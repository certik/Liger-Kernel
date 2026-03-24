"""
Tokenizers-lite: minimal shim for the HuggingFace tokenizers package.

Only implements enough stubs for test/utils.py imports.
"""


class AddedToken:
    """Stub for tokenizers.AddedToken."""

    def __init__(self, content="", single_word=False, lstrip=False, rstrip=False, normalized=True):
        self.content = content
        self.single_word = single_word
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.normalized = normalized

    def __repr__(self):
        return f"AddedToken({self.content!r})"


class Tokenizer:
    """Stub for tokenizers.Tokenizer."""

    def __init__(self, model=None):
        self.model = model
        self._pre_tokenizer = None

    @property
    def pre_tokenizer(self):
        return self._pre_tokenizer

    @pre_tokenizer.setter
    def pre_tokenizer(self, value):
        self._pre_tokenizer = value

    def train_from_iterator(self, iterator, trainer=None):
        pass

    def encode(self, text):
        return type("Encoding", (), {"ids": [], "tokens": []})()
