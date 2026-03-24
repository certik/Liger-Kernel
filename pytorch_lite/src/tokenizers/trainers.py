"""Stub for tokenizers.trainers."""


class BpeTrainer:
    """Stub for tokenizers.trainers.BpeTrainer."""

    def __init__(self, vocab_size=30000, special_tokens=None, **kwargs):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
