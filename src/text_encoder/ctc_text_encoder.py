import re
from string import ascii_lowercase

import torch
from torchaudio.models.decoder import ctc_decoder as torch_ctc_decoder
from torchaudio.models.decoder import download_pretrained_files

files = download_pretrained_files("librispeech-4-gram")

# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
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
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = ""
        last_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind == last_ind:
                continue
            if ind != self.char2ind[self.EMPTY_TOK]:
                decoded += self.ind2char[ind]
            last_ind = ind

        return decoded

    def ctc_decode_beam_search(self, probs) -> str:
        tokens = list(self.char2ind.keys())
        beam_search = torch_ctc_decoder(
            lexicon=files.lexicon[:-4] + "_processed.txt",
            tokens=tokens,
            blank_token="",
            sil_token=" ",
            lm=files.lm,
        )
        inds = beam_search(probs)
        ind_decoded = [" ".join(hypo[0].words) for hypo in inds]

        return ind_decoded

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
