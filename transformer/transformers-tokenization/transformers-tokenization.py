import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # Clear existing vocab
        self.word_to_id = {}
        self.id_to_word = {}
        
        # Add special tokens first
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token in special_tokens:
            self.word_to_id[token] = len(self.word_to_id)
        
        # Extract unique words from all texts (lowercased)
        unique_words = set()
        for text in texts:
            words = text.lower().split()
            unique_words.update(words)
        
        # Add unique words to vocabulary (sorted for determinism)
        for word in sorted(unique_words):
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.word_to_id)
        
        # Build reverse mapping
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().split()
        ids = []
        for word in words:
            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                ids.append(self.word_to_id[self.unk_token])
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for idx in ids:
            if idx in self.id_to_word:
                words.append(self.id_to_word[idx])
            else:
                words.append(self.unk_token)
        return " ".join(words)
