import logging
import torch


logger = logging.getLogger(__name__)


class DataLoaderLite:
    def __init__(self, B, T, text, tokenizer):
        self.B = B
        self.T = T
        self.text = text
        self.tokenizer = tokenizer

        # at init load tokens from disk and store them in memory
        tokens = tokenizer.encode(text).ids
        self.tokens = torch.tensor(tokens)
        logger.info("loaded %s tokens", len(self.tokens))
        self.batch_per_epoch = len(self.tokens) // (B * T)
        logger.info("1 epoch = %s batches", self.batch_per_epoch)

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
