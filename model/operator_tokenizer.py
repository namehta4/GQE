"""
Custom word-level tokenizer for ADAPT-VQE operator sequences, reproducing
Section 3.2.1 of ADAPT-GQE (arXiv:2607.22468): one token per operator-pool
index, per-digit tokenization for numeric coefficients (0-9, '.', '-'), plus
a <ham> token used to inject the multimodal Hamiltonian embedding.

The vocabulary is specific to ONE (active space, operator pool) combination
-- a tokenizer built with pool_size=500 is NOT interchangeable with a
different pool size or a different pool type (UCCSD vs UCCGSD), matching the
paper's design of training independent models per qubit count (end of
Sec. 3.2).

Not yet exercised against a live `transformers` install in this environment
-- the PreTrainedTokenizer base-class integration (special-token handling
order in particular) should be smoke-tested on a short example string
before relying on it for a full training run:

    tok = OperatorTokenizer(pool_size=500)
    ids = tok.encode("<bos><op253>-0.054371<ham><op17>0.001783<ham><eos>")
    assert tok.decode(ids) produces back the same structure.
"""
import json
import os
import re
from typing import Optional

from transformers import PreTrainedTokenizer


class OperatorTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}

    _TOKEN_RE = re.compile(r"<[^>]+>|.")

    def __init__(
        self,
        pool_size: Optional[int] = None,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        ham_token="<ham>",
        vocab_file: Optional[str] = None,
        **kwargs,
    ):
        self.ham_token = ham_token

        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file) as f:
                self._vocab = json.load(f)
            # Recover pool_size from the loaded vocab if not explicitly given.
            n_op_tokens = sum(1 for k in self._vocab if k.startswith("<op"))
            self.pool_size = pool_size if pool_size is not None else n_op_tokens
        else:
            if pool_size is None:
                raise ValueError(
                    "pool_size is required when constructing a new tokenizer "
                    "(no existing vocab_file was provided)"
                )
            self.pool_size = pool_size
            self._vocab = self._build_vocab(
                pool_size, pad_token, bos_token, eos_token, unk_token, ham_token
            )

        self._id_to_token = {v: k for k, v in self._vocab.items()}

        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            additional_special_tokens=[ham_token],
            **kwargs,
        )

    @staticmethod
    def _build_vocab(pool_size, pad_token, bos_token, eos_token, unk_token, ham_token):
        vocab = {}
        for tok in (pad_token, bos_token, eos_token, unk_token, ham_token):
            vocab[tok] = len(vocab)
        for ch in "0123456789.-":
            vocab[ch] = len(vocab)
        for i in range(pool_size):
            vocab[f"<op{i}>"] = len(vocab)
        return vocab

    @property
    def vocab_size(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text, **kwargs):
        return self._TOKEN_RE.findall(text)

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, self._vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        return self._id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        path = os.path.join(save_directory, f"{prefix}vocab.json")
        with open(path, "w") as f:
            json.dump(self._vocab, f)
        return (path,)

    @property
    def ham_token_id(self):
        return self._vocab[self.ham_token]

    def build_sequence_text(self, operator_sequence):
        """operator_sequence: list of (pool_index, coefficient) pairs, as
        produced by chemistry/run_adapt_vqe.py. Places the <ham> token after
        EACH operator-coefficient pair, per Sec. 3.2.2."""
        parts = [f"<op{idx}>{coeff:.6f}{self.ham_token}" for idx, coeff in operator_sequence]
        return f"{self.bos_token}{''.join(parts)}{self.eos_token}"
