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
        **kwargs
    ):
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
        
        # Special token IDs
        self.start_id = card_to_index[deck_config.start_token]
        self.end_id = card_to_index[deck_config.end_token]
        self.pad_id = card_to_index[deck_config.pad_token]
        self.special_ids = tf.constant([self.start_id, self.end_id, self.pad_id], dtype=tf.int32)
        self.max_length = deck_config.max_total_cards + 2
        
    def call(self, inputs, training=False):
        """Forward pass through base model."""
        return self.base_model(inputs, training=training)
    
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
                    card_feature_inputs = []
                    for feat in x[2:]:
                        feat_shape = tf.shape(feat)
                        rank = tf.rank(feat)
                        zero = tf.constant(0, dtype=feat_shape.dtype)
                        batch_dim = tf.where(rank > 0, feat_shape[0], zero)
                        broadcasted = tf.logical_or(tf.equal(batch_dim, 0), tf.equal(batch_dim, 1))
                        def slice_feat():
                            slice_sizes = tf.concat([[num_generate], tf.fill([rank - 1], -1)], axis=0)
                            return tf.slice(feat, tf.zeros_like(slice_sizes), slice_sizes)
                        sliced = tf.cond(
                            broadcasted,
                            lambda: feat,
                            lambda: slice_feat()
                        )
                        card_feature_inputs.append(sliced)
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
        
        # Update metrics
        self.base_model.compiled_metrics.update_state(y, outputs)
        
        # Return metrics
        metrics = {m.name: m.result() for m in self.base_model.metrics}
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
        
        # Prepare card features if available
        model_inputs_base = [prompt_tokens]
        if self.use_card_features and card_feature_inputs:
            # Card features are constant tensors that broadcast to batch size
            # They're typically (1, vocab_size) or (1, vocab_size, embed_dim) for ability
            # TensorFlow will automatically broadcast them, so we can use as-is
            model_inputs_base.extend(card_feature_inputs)
        
        # Generate autoregressively
        for step in range(max_length - 1):
            # Prepare decoder input (current sequence)
            current_length = tf.shape(generated)[1]
            decoder_input = generated
            
            # Pad to max_length if needed (though we'll stop at max_length)
            if current_length < max_length:
                padding = tf.fill(
                    [batch_size, max_length - current_length],
                    self.pad_id
                )
                decoder_input = tf.concat([decoder_input, padding], axis=1)
            
            # Build model inputs
            model_inputs = model_inputs_base + [decoder_input]
            
            # Forward pass
            outputs = self.base_model(model_inputs, training=training)
            
            # Extract main logits
            main_logits = outputs["main"]  # (batch, seq_len, vocab_size)
            
            # Get logits for the next token (at position current_length - 1)
            next_token_logits = main_logits[:, current_length - 1, :]  # (batch, vocab_size)
            
            # Sample next token (use scheduled sampling: mix teacher forcing with model predictions)
            if training and tf.random.uniform([]) < self.scheduled_sampling_rate:
                # Use model's own predictions (autoregressive)
                # Apply temperature and sample
                temperature = 0.8
                next_token_logits = next_token_logits / temperature
                next_token = tf.random.categorical(
                    next_token_logits,
                    num_samples=1,
                    dtype=tf.int32
                )  # (batch, 1)
                next_token = tf.squeeze(next_token, axis=1)  # (batch,)
            else:
                # Use argmax (greedy) for efficiency
                next_token = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)  # (batch,)
            
            # Check for EOS
            eos_mask = tf.equal(next_token, self.end_id)
            if tf.reduce_all(eos_mask):
                # All sequences ended
                break
            
            # Append next token
            next_token = tf.expand_dims(next_token, axis=1)  # (batch, 1)
            generated = tf.concat([generated, next_token], axis=1)
            
            # Stop if we've reached max length
            if tf.shape(generated)[1] >= max_length:
                break
        
        # Pad to max_length
        current_length = tf.shape(generated)[1]
        if current_length < max_length:
            padding = tf.fill(
                [batch_size, max_length - current_length],
                self.pad_id
            )
            generated = tf.concat([generated, padding], axis=1)
        
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
    
    def test_step(self, data):
        """Standard test step (no autoregressive generation for efficiency)."""
        x, y = data
        outputs = self.base_model(x, training=False)
        
        # Update metrics
        self.base_model.compiled_metrics.update_state(y, outputs)
        
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
        
        metrics = {m.name: m.result() for m in self.base_model.metrics}
        metrics.update(loss_values)
        metrics["loss"] = total_loss
        
        return metrics

