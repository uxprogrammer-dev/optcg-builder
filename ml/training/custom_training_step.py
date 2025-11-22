"""
Phase 2: Custom training step with autoregressive sequence-level loss.

This implements true autoregressive generation during training, not just argmax
from teacher forcing. This directly penalizes singleton-heavy generated sequences.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import keras

from ..config import DeckConfig


class AutoregressiveSequenceLossStep(keras.Model):
    """
    Custom training step that generates sequences autoregressively during training
    and computes sequence-level loss from the generated sequences.
    
    This is Phase 2 of Solution 1: directly penalize singleton-heavy generations
    by computing loss from actual autoregressive generation, not teacher forcing.
    """
    
    def __init__(
        self,
        base_model: keras.Model,
        sequence_level_loss_fn,
        sequence_level_weight: float,
        card_to_index: Dict[str, int],
        index_to_card: Dict[int, str],
        deck_config: DeckConfig = DeckConfig(),
        use_card_features: bool = True,
        scheduled_sampling_rate: float = 0.5,
        generation_batch_fraction: float = 0.25,  # Only generate for 25% of batch to save compute
        losses: Optional[Dict] = None,
        loss_weights: Optional[Dict] = None,
        max_generate_length: Optional[int] = None,
        **kwargs
    ):
        # Backward compatibility: older call sites passed metric_output_names, ignore them
        kwargs.pop("metric_output_names", None)
        super().__init__(**kwargs)
        self.base_model = base_model
        self.sequence_level_loss_fn = sequence_level_loss_fn
        self.sequence_level_weight = sequence_level_weight
        self.card_to_index = card_to_index
        self.index_to_card = index_to_card
        self.deck_config = deck_config
        self.use_card_features = use_card_features
        self.scheduled_sampling_rate = scheduled_sampling_rate
        self.generation_batch_fraction = generation_batch_fraction
        # Store losses and loss weights for manual computation
        self.losses_dict = losses or {}
        self.loss_weights_dict = loss_weights or {}
        self.max_generate_length = max_generate_length
        
        # Special token IDs
        self.start_id = card_to_index[deck_config.start_token]
        self.end_id = card_to_index[deck_config.end_token]
        self.pad_id = card_to_index[deck_config.pad_token]
        self.special_ids = tf.constant([self.start_id, self.end_id, self.pad_id], dtype=tf.int32)
        default_max = deck_config.max_total_cards + 2
        if self.max_generate_length is not None:
            self.max_length = max(2, int(self.max_generate_length))
        else:
            self.max_length = default_max
        self.output_names = list(base_model.output_names)
        
    def call(self, inputs, training=False):
        """Forward pass through base model."""
        return self.base_model(inputs, training=training)
    
    @property
    def metrics(self):
        # Share base model metrics so Keras logging/reset works unchanged
        return self.base_model.metrics
    
    def train_step(self, data):
        """
        Custom training step that:
        1. Performs normal forward pass with teacher forcing
        2. Generates sequences autoregressively for a subset of the batch
        3. Computes sequence-level loss from generated sequences
        4. Combines with main losses
        """
        # Unpack data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None
        
        # Normal forward pass with teacher forcing (efficient)
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.base_model(x, training=True)
            
            # Compute main losses (main, type_aux, cost_aux, freq_hist)
            total_loss = 0.0
            loss_values = {}
            
            # Get loss functions and weights from stored dicts
            loss_fns = self.losses_dict
            loss_weights = self.loss_weights_dict
            
            for output_name, target in y.items():
                if output_name == "predicted_sequence_freq_hist":
                    # Skip this - we'll compute it from generated sequences
                    continue
                
                if output_name in loss_fns:
                    loss_fn = loss_fns[output_name]
                    loss_weight = loss_weights.get(output_name, 1.0)
                    
                    if output_name in outputs:
                        output = outputs[output_name]
                        loss_value = loss_fn(target, output)
                        weighted_loss = loss_weight * loss_value
                        total_loss += weighted_loss
                        loss_values[f"{output_name}_loss"] = loss_value
            
            # Phase 2: Generate sequences autoregressively and compute sequence-level loss
            batch_size = tf.shape(x[0])[0] if isinstance(x, (list, tuple)) else tf.shape(x)[0]
            batch_size_float = tf.cast(batch_size, tf.float32)
            generation_fraction = tf.constant(self.generation_batch_fraction, dtype=tf.float32)
            num_generate = tf.cast(
                tf.math.ceil(batch_size_float * generation_fraction),
                tf.int32
            )
            num_generate = tf.maximum(num_generate, 1)  # Ensure at least 1 sample for generation
            
            # Extract prompt tokens and card features from inputs for autoregressive generation
            if isinstance(x, (list, tuple)):
                prompt_tokens = x[0]
                if self.use_card_features and len(x) > 2:
                    # Card features are global tensors (shape (1, vocab_size, ...)).
                    # They broadcast automatically for any batch size, so we can pass them as-is.
                    card_feature_inputs = list(x[2:])
                else:
                    card_feature_inputs = []
            else:
                prompt_tokens = x
                card_feature_inputs = []
            
            generated_sequences = self._generate_autoregressive_batch(
                prompt_tokens[:num_generate],
                card_feature_inputs,
                training=True
            )
            
            generated_freq_hist = self._compute_freq_hist_from_sequences(
                generated_sequences
            )
            
            target_freq_hist = y.get("predicted_sequence_freq_hist")
            if target_freq_hist is not None:
                target_freq_hist_subset = target_freq_hist[:num_generate]
                sequence_loss = self.sequence_level_loss_fn(
                    target_freq_hist_subset,
                    generated_freq_hist
                )
                weighted_sequence_loss = self.sequence_level_weight * sequence_loss
                total_loss += weighted_sequence_loss
                loss_values["autoregressive_sequence_loss"] = sequence_loss
        
        # Compute gradients and update weights
        trainable_vars = self.base_model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Apply gradients
        self.base_model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics using base model's compiled metrics (only for outputs that have metrics)
        self._update_metrics(y, outputs)
        
        # Return metrics
        metrics = self._collect_metric_results()
        metrics.update(loss_values)
        metrics["loss"] = total_loss
        
        return metrics
    
    def _generate_autoregressive_batch(
        self,
        prompt_tokens: tf.Tensor,
        card_feature_inputs: list,
        training: bool = True
    ) -> tf.Tensor:
        """
        Generate sequences autoregressively for a batch of prompts.
        
        Returns:
            generated_sequences: (batch, max_length) tensor of token IDs
        """
        batch_size = tf.shape(prompt_tokens)[0]
        max_length = self.max_length
        
        # Initialize with BOS token
        generated = tf.fill([batch_size, 1], self.start_id)  # (batch, 1)
        
        # Prepare card features if available (stored separately to maintain input order)
        model_inputs_base = [prompt_tokens]
        feature_inputs = list(card_feature_inputs) if (self.use_card_features and card_feature_inputs) else []
        
        finished = tf.zeros([batch_size], dtype=tf.bool)
        training_flag = tf.convert_to_tensor(training, dtype=tf.bool)
        temperature = tf.constant(0.8, dtype=tf.float32)
        
        for _ in range(max_length - 1):
            current_length = tf.shape(generated)[1]
            pad_len = tf.maximum(max_length - current_length, 0)
            padding = tf.fill(tf.stack([batch_size, pad_len]), self.pad_id)
            decoder_input = tf.concat([generated, padding], axis=1)
            decoder_input = decoder_input[:, :max_length]
            
            model_inputs = [model_inputs_base[0], decoder_input]
            if feature_inputs:
                model_inputs.extend(feature_inputs)
            outputs = self.base_model(model_inputs, training=training)
            main_logits = outputs["main"]
            logit_position = current_length - 1
            next_token_logits = main_logits[:, logit_position, :]
            
            random_val = tf.random.uniform([], dtype=tf.float32)
            sample_condition = tf.logical_and(
                training_flag,
                tf.less(random_val, self.scheduled_sampling_rate)
            )
            
            def sample_next():
                sampled = tf.random.categorical(
                    next_token_logits / temperature,
                    num_samples=1,
                    dtype=tf.int32
                )
                return tf.squeeze(sampled, axis=1)
            
            def greedy_next():
                return tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
            
            next_token_candidate = tf.cond(sample_condition, sample_next, greedy_next)
            write_token = tf.where(
                finished,
                tf.fill([batch_size], self.pad_id),
                next_token_candidate
            )
            generated = tf.concat([generated, tf.expand_dims(write_token, axis=1)], axis=1)
            finished = tf.logical_or(finished, tf.equal(write_token, self.end_id))
        
        current_length = tf.shape(generated)[1]
        pad_len = tf.maximum(max_length - current_length, 0)
        padding = tf.fill(tf.stack([batch_size, pad_len]), self.pad_id)
        generated = tf.concat([generated, padding], axis=1)
        generated = generated[:, :max_length]
        
        return generated  # (batch, max_length)
    
    def _compute_freq_hist_from_sequences(
        self,
        sequences: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute frequency histogram from generated sequences.
        
        Args:
            sequences: (batch, seq_len) tensor of token IDs
        
        Returns:
            freq_hist: (batch, vocab_size) tensor of normalized frequencies [0, 1]
        """
        batch_size = tf.shape(sequences)[0]
        vocab_size = len(self.card_to_index)
        
        # One-hot encode sequences
        sequences_one_hot = tf.one_hot(
            sequences,
            depth=vocab_size,
            dtype=tf.float32
        )  # (batch, seq_len, vocab_size)
        
        # Sum across sequence length to get frequency counts
        freq_counts = tf.reduce_sum(sequences_one_hot, axis=1)  # (batch, vocab_size)
        
        # Mask out special tokens
        special_mask = tf.ones((vocab_size,), dtype=tf.float32)
        special_mask = tf.tensor_scatter_nd_update(
            special_mask,
            tf.expand_dims(self.special_ids, 1),
            tf.zeros((tf.shape(self.special_ids)[0],), dtype=tf.float32)
        )
        special_mask = tf.expand_dims(special_mask, 0)  # (1, vocab_size)
        freq_counts = freq_counts * special_mask
        
        # Cap at max_copies and normalize
        max_copies = tf.cast(self.deck_config.max_copies_per_card, tf.float32)
        freq_counts = tf.minimum(freq_counts, max_copies)
        freq_hist = tf.math.divide_no_nan(freq_counts, max_copies)  # (batch, vocab_size)
        
        return freq_hist
    
    def _update_metrics(self, y_dict, outputs_dict):
        for metric in self.base_model.metrics:
            name = metric.name or ""
            if name.startswith("main") and "main" in y_dict and "main" in outputs_dict:
                metric.update_state(y_dict["main"], outputs_dict["main"])
            elif name.startswith("freq_hist") and "freq_hist" in y_dict and "freq_hist" in outputs_dict:
                metric.update_state(y_dict["freq_hist"], outputs_dict["freq_hist"])

    def _collect_metric_results(self):
        metric_values = {}
        for metric in self.base_model.metrics:
            try:
                metric_values[metric.name] = metric.result()
            except ValueError:
                continue
        return metric_values
    
    def test_step(self, data):
        """Standard test step (no autoregressive generation for efficiency)."""
        x, y = data
        outputs = self.base_model(x, training=False)
        
        # Update metrics
        self._update_metrics(y, outputs)
        
        # Compute losses
        loss_values = {}
        total_loss = 0.0
        
        loss_fns = self.losses_dict
        loss_weights = self.loss_weights_dict
        
        for output_name, target in y.items():
            if output_name == "predicted_sequence_freq_hist":
                continue
            
            if output_name in loss_fns:
                loss_fn = loss_fns[output_name]
                loss_weight = loss_weights.get(output_name, 1.0)
                
                if output_name in outputs:
                    output = outputs[output_name]
                    loss_value = loss_fn(target, output)
                    weighted_loss = loss_weight * loss_value
                    total_loss += weighted_loss
                    loss_values[f"{output_name}_loss"] = loss_value
        
        metrics = self._collect_metric_results()
        metrics.update(loss_values)
        metrics["loss"] = total_loss
        
        return metrics

