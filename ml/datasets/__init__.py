"""
Dataset builders for prompt→deck training.
"""

from .synthetic import SyntheticDeckGenerator, generate_synthetic_examples, write_examples_jsonl, load_examples_jsonl
from .tournament import load_tournament_examples, TournamentDeck
from .rules import load_rules_summary

__all__ = [
    "SyntheticDeckGenerator",
    "generate_synthetic_examples",
    "write_examples_jsonl",
    "load_examples_jsonl",
    "load_tournament_examples",
    "TournamentDeck",
    "load_rules_summary",
]

try:  # Card features require TensorFlow
    from .card_features import (
        build_card_feature_matrix,
        create_ability_vectorizer,
        encode_card_features,
    )
    __all__.extend([
        "build_card_feature_matrix",
        "create_ability_vectorizer",
        "encode_card_features",
    ])
except ModuleNotFoundError:
    pass

try:  # TensorFlow may not be installed when only data generation is required.
    from .tfdata import (
        build_card_vocabulary,
        create_text_vectorizer,
        create_text_vectorizer_from_vocab,
        decode_card_sequence,
        make_tf_dataset,
    )

    __all__.extend(
        [
            "build_card_vocabulary",
            "create_text_vectorizer",
            "create_text_vectorizer_from_vocab",
            "decode_card_sequence",
            "make_tf_dataset",
        ]
    )
except ModuleNotFoundError:
    pass

