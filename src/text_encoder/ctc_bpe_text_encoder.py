import re
from string import ascii_lowercase

import numpy as np
import sentencepiece as spm
import torch
from torchaudio.models.decoder import ctc_decoder as torch_ctc_decoder
from torchaudio.models.decoder import download_pretrained_files

lm = download_pretrained_files(
    "librispeech-4-gram"
)  # "librispeech" "librispeech-3-gram", "librispeech-4-gram" and "librispeech".

# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCBPETextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        f = open("bpe_vocab.vc")

        line = f.readline()

        self.vocab = line.split(",")
        self.vocab[-1] = "q"

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        print(10 * "!!!!!")
        print(self.vocab)
        print(self.char2ind)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char.decode(item)

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            inds = []
            cur_str = ""
            for sym in text:
                cur_str += sym
                if cur_str not in self.char2ind:
                    inds.append(self.char2ind[cur_str[:-1]])
                    cur_str = cur_str[-1]
            inds.append(self.char2ind[cur_str])
            return torch.Tensor(inds).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join(self.tokenizer.Decode(inds)).strip()

    def ctc_decode(self, inds) -> str:
        # print(inds)

        decoded = ""
        last_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind == last_ind:
                continue
            if ind != self.char2ind[self.EMPTY_TOK]:
                decoded += self.ind2char[ind]
            last_ind = ind

        return decoded

    def ctc_decode_custome_beam_search(self, probs, num_hypos=1) -> str:
        pass

    def ctc_decode_beam_search(self, probs) -> str:
        pass

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
