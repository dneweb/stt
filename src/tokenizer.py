"""
Multilingual Character-level Tokenizer for English, Hindi, and Gujarati.
Supports CTC decoding, special tokens (<BLANK>, <PAD>, <UNK>), and Unicode NFKC normalization.
"""

import json
import unicodedata
from pathlib import Path
from typing import List, Union


class CharTokenizer:
    """
    Character-level tokenizer loaded from vocab.json.
    Covers Latin, Devanagari, Gujarati, punctuation, and CTC special tokens.
    """

    def __init__(self, vocab_path: Union[str, Path]):
        self.vocab_path = Path(vocab_path)
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found at: {self.vocab_path}")

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.char_to_id = data.get("char_to_id", {})
        # Ensure integer keys for id_to_char
        raw_id_to_char = data.get("id_to_char", {})
        self.id_to_char = {int(k): v for k, v in raw_id_to_char.items()}

        # Reconstruct missing reverse mappings if needed
        for ch, idx in self.char_to_id.items():
            self.id_to_char[idx] = ch

        # Special token IDs
        self.blank_id = self.char_to_id.get("<BLANK>", 0)
        self.pad_id = self.char_to_id.get("<PAD>", 1)
        self.sos_id = self.char_to_id.get("<SOS>", 2)
        self.eos_id = self.char_to_id.get("<EOS>", 3)
        self.unk_id = self.char_to_id.get("<UNK>", 4)
        self.space_id = self.char_to_id.get(" ", 5)

        self.special_ids = {self.blank_id, self.pad_id, self.sos_id, self.eos_id, self.unk_id}

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def normalize_text(self, text: str) -> str:
        """Applies Unicode NFKC canonical normalization and cleans whitespace."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        return " ".join(text.split()).strip()

    def encode(self, text: str) -> List[int]:
        """Converts text string into a list of token IDs."""
        text = self.normalize_text(text)
        token_ids = []
        for ch in text:
            token_ids.append(self.char_to_id.get(ch, self.unk_id))
        return token_ids

    def decode(self, ids: List[int], remove_special: bool = True) -> str:
        """Converts token IDs back into string."""
        chars = []
        for idx in ids:
            idx = int(idx)
            if remove_special and idx in self.special_ids:
                continue
            chars.append(self.id_to_char.get(idx, ""))
        return "".join(chars)

    def ctc_decode(self, ids: List[int]) -> str:
        """
        Greedy CTC decoding:
        1. Collapse consecutive identical tokens.
        2. Remove CTC blank tokens (ID 0) and pad tokens (ID 1).
        3. Convert remaining IDs to text.
        """
        collapsed = []
        prev_id = None

        for raw_id in ids:
            idx = int(raw_id)
            if idx != prev_id:
                if idx != self.blank_id and idx != self.pad_id:
                    collapsed.append(idx)
                prev_id = idx

        return self.decode(collapsed, remove_special=True)

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size}, blank_id={self.blank_id}, pad_id={self.pad_id})"
