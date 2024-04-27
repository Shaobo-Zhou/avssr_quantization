import multiprocessing
import re
from string import ascii_lowercase

import torch
from pyctcdecode import build_ctcdecoder

from src.utils.io_utils import ROOT_PATH


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, use_lm=False, **kwargs):
        """
        :param alphabet: alphabet for language. If None it will be set to ascii
        :param use_lm: whether to enable the support for Language Model Beam Search
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.bs_decoder = self._create_bs_decoder()

        self.use_lm = use_lm
        if use_lm:
            self.kenlm_path = ROOT_PATH / "data" / "lm" / "librispeech" / "4-gram.arpa"
            self.unigram_path = self.kenlm_path.parent / "librispeech-vocab.txt"
            self.lm_decoder = self._create_lm_decoder()

    def __len__(self):
        return len(self.vocab)

    def _create_bs_decoder(self):
        """
        Creates beam search ctc_decoder
        """
        vocab = self.vocab

        vocab = [elem.upper() for elem in vocab]

        decoder = build_ctcdecoder(vocab)

        return decoder

    def _create_lm_decoder(self):
        """
        Creates ctc decoder with the support of Language Model
        """
        vocab = self.vocab

        vocab = [elem.upper() for elem in vocab]

        with self.unigram_path.open() as f:
            unigram_list = [t for t in f.read().strip().split("\n")]

        decoder = build_ctcdecoder(
            vocab, kenlm_model_path=str(self.kenlm_path), unigrams=unigram_list
        )

        return decoder

    def encode(self, text):
        """
        Encoding of the text
        :param text: text which will be encoded

        Works for both BPE and usual alphabet
        """
        text = self.normalize_text(text)
        return torch.tensor([self.char2ind[char] for char in text])

    def ctc_decode(self, inds) -> str:
        """
        CTC decoding: list of int to text
        :param inds: list of indecies (encoded text)
        """
        last_char = self.EMPTY_TOK
        text = []
        for ind in inds:
            if self.ind2char[ind] == last_char:
                continue
            char = self.ind2char[ind]
            if char != self.EMPTY_TOK:
                text.append(char)
            last_char = char
        return "".join(text).strip()

    def ctc_argmax(self, probs: torch.tensor, lengths: torch.tensor):
        """
        Performs argmax ctc decode and returns
        lists of decoded texts.
        :param probs: probabilities from model with shape [N, L, H]
        :param lengths: tensor of shape [N,] containing lengthes without padding
        """

        probs = torch.nn.functional.log_softmax(probs.detach().cpu(), -1)  # to be sure
        logits_list = [probs[i][: lengths[i]].numpy() for i in range(lengths.shape[0])]

        text_list = []
        for logits in logits_list:
            text_list.append(self.ctc_decode(logits.argmax(-1)))
        return text_list

    def ctc_beam_search(
        self,
        probs: torch.tensor,
        lengths: torch.tensor,
        beam_size: int = 100,
        use_lm=False,
    ):
        """
        Performs beam search (with or without language model) and returns
        list of decoded texts.
        :param probs: probabilities from model with shape [N, L, H]
        :param lengths: tensor of shape [N,] containing lengthes without padding
        :param beam_size: size of beam to use in decoding
        :param use_lm: use lm version or classic bs
        """
        if use_lm:
            decoder = self.lm_decoder
        else:
            decoder = self.bs_decoder

        probs = torch.nn.functional.log_softmax(probs.detach().cpu(), -1)  # to be sure
        logits_list = [probs[i][: lengths[i]].numpy() for i in range(lengths.shape[0])]

        with multiprocessing.get_context("fork").Pool() as pool:
            text_list = decoder.decode_batch(pool, logits_list, beam_width=beam_size)

        text_list = [text.strip().lower() for text in text_list]

        return text_list

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
