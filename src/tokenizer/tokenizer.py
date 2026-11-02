"""
Multilingual Character Tokenizer — Maps characters to integer token IDs.

Supports multilingual text (English, Hindi, Gujarati) with special tokens
for CTC blank, padding, start-of-sequence, end-of-sequence, and unknown tokens.

"""

import json
import os
import csv
from typing import Dict, List, Optional, Union


class CharTokenizer:
    """Character-level tokenizer for multilingual STT (English, Hindi, Gujarati).

    Attributes:
        blank_token: CTC blank token string.
        pad_token: Padding token string.
        sos_token: Start-of-sequence token string.
        eos_token: End-of-sequence token string.
        unk_token: Unknown character token string.
        char_to_id: Mapping from character string to integer ID.
        id_to_char: Mapping from integer ID to character string.
    """

    SPECIAL_TOKENS = ["<BLANK>", "<PAD>", "<SOS>", "<EOS>", "<UNK>"]

    def __init__(
        self,
        blank_token: str = "<BLANK>",
        pad_token: str = "<PAD>",
        sos_token: str = "<SOS>",
        eos_token: str = "<EOS>",
        unk_token: str = "<UNK>",
    ) -> None:
        self.blank_token = blank_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self._initialized = False

        # Pre-assign special tokens
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Initialize special tokens at the beginning of the vocabulary."""
        special_list = [
            self.blank_token,
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
        ]
        self.char_to_id = {token: idx for idx, token in enumerate(special_list)}
        self.id_to_char = {idx: token for idx, token in enumerate(special_list)}

    @property
    def blank_id(self) -> int:
        return self.char_to_id[self.blank_token]

    @property
    def pad_id(self) -> int:
        return self.char_to_id[self.pad_token]

    @property
    def sos_id(self) -> int:
        return self.char_to_id[self.sos_token]

    @property
    def eos_id(self) -> int:
        return self.char_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.char_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def build_vocab(self, texts: List[str]) -> None:
        """Build character vocabulary from a list of text strings.

        Args:
            texts: List of transcription strings to extract characters from.
        """
        self._init_special_tokens()
        next_id = len(self.char_to_id)

        # Extract all unique characters
        unique_chars = set()
        for text in texts:
            if text:
                unique_chars.update(list(text))

        # Sort characters deterministically: ASCII first, then Unicode by code point
        sorted_chars = sorted(list(unique_chars))

        for char in sorted_chars:
            if char not in self.char_to_id:
                self.char_to_id[char] = next_id
                self.id_to_char[next_id] = char
                next_id += 1

        self._initialized = True

    def build_from_metadata(self, csv_path: str = "data/metadata/metadata.csv") -> None:
        """Build vocabulary directly from the project's metadata.csv file.

        Args:
            csv_path: Path to the metadata CSV file.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        texts = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                transcript = row.get("transcript", "")
                if transcript:
                    texts.append(transcript)

        self.build_vocab(texts)

    def encode(
        self,
        text: str,
        add_sos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Convert a text string into a list of token IDs.

        Args:
            text: Input transcription text.
            add_sos: Whether to prepend <SOS> token ID.
            add_eos: Whether to append <EOS> token ID.

        Returns:
            List of integer token IDs.
        """
        unk_id = self.unk_id
        token_ids = []

        if add_sos:
            token_ids.append(self.sos_id)

        for char in text:
            token_ids.append(self.char_to_id.get(char, unk_id))

        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(
        self,
        token_ids: List[int],
        remove_special: bool = True,
    ) -> str:
        """Convert a list of token IDs back into a text string.

        Args:
            token_ids: Sequence of integer token IDs.
            remove_special: Whether to filter out special tokens (<PAD>, <SOS>, etc.).

        Returns:
            Decoded text string.
        """
        special_ids = {self.blank_id, self.pad_id, self.sos_id, self.eos_id, self.unk_id}
        chars = []

        for tid in token_ids:
            if remove_special and tid in special_ids:
                continue
            char = self.id_to_char.get(tid, "")
            chars.append(char)

        return "".join(chars)

    def decode_ctc(self, token_ids: List[int]) -> str:
        """Apply CTC greedy decoding (collapses repeats and removes blanks).

        Args:
            token_ids: Raw frame-level predicted token IDs.

        Returns:
            Decoded transcription string.
        """
        blank = self.blank_id
        collapsed = []
        prev = None

        for tid in token_ids:
            if tid != prev:
                if tid != blank:
                    collapsed.append(tid)
                prev = tid

        return self.decode(collapsed, remove_special=True)

    def save_vocab(self, file_path: str = "tokenizer/vocab.json") -> None:
        """Save vocabulary mapping to a JSON file.

        Args:
            file_path: Output file path.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data = {
            "char_to_id": self.char_to_id,
            "special_tokens": {
                "blank": self.blank_token,
                "pad": self.pad_token,
                "sos": self.sos_token,
                "eos": self.eos_token,
                "unk": self.unk_token,
            },
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, file_path: str = "tokenizer/vocab.json") -> None:
        """Load vocabulary mapping from a JSON file.

        Args:
            file_path: Path to the vocab JSON file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Vocab file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self._initialized = True


if __name__ == "__main__":
    print("=" * 60)
    print("CharTokenizer — Self Verification Test")
    print("=" * 60)

    tokenizer = CharTokenizer()
    metadata_file = "data/metadata/metadata.csv"

    if os.path.exists(metadata_file):
        tokenizer.build_from_metadata(metadata_file)
        vocab_path = "tokenizer/vocab.json"
        tokenizer.save_vocab(vocab_path)
        print(f"✓ Built vocabulary from {metadata_file}")
        print(f"  Vocabulary size: {tokenizer.vocab_size} tokens")
        print(f"  Saved vocab to : {vocab_path}")

        # Test encoding and decoding in all 3 languages
        samples = [
            ("English", "Water. Table. Morning. Hello world!"),
            ("Hindi", "किताब. कमरा. बारिश. आज दिन कैसा रहा?"),
            ("Gujarati", "AnyDesk બંધ કરો. પાણી. ઘર. ચા."),
        ]

        print("\n--- Encode / Decode Verification ---")
        for lang, text in samples:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            match = "✓ MATCH" if text == decoded else "❌ MISMATCH"
            print(f"[{lang}] {match}")
            print(f"  Original: {text}")
            print(f"  Encoded : {encoded[:10]}... (total {len(encoded)} tokens)")
            print(f"  Decoded : {decoded}")

        # Test CTC Decoding
        ctc_raw_tokens = [
            tokenizer.blank_id,
            tokenizer.char_to_id.get("W", 0),
            tokenizer.char_to_id.get("W", 0),  # repeated
            tokenizer.blank_id,
            tokenizer.char_to_id.get("a", 0),
            tokenizer.char_to_id.get("t", 0),
            tokenizer.char_to_id.get("e", 0),
            tokenizer.char_to_id.get("r", 0),
            tokenizer.blank_id,
        ]
        ctc_decoded = tokenizer.decode_ctc(ctc_raw_tokens)
        print(f"\n[CTC Decoding Test]")
        print(f"  Raw tokens: {ctc_raw_tokens}")
        print(f"  Decoded   : '{ctc_decoded}' (Expected: 'Water')")
        assert ctc_decoded == "Water", f"CTC decode failed: got {ctc_decoded}"
        print("  ✓ CTC Decoding verified successfully!")

    print("=" * 60)
