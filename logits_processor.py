import torch
from transformers import LogitsProcessor

class MaskNonInputLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` that enforces the logits does not 
    """

    def __init__(self, decoder_token_masks: torch.BoolTensor):
        

        self.decoder_token_masks = decoder_token_masks

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        
        beam_size = scores.size(0) // self.decoder_token_masks.size(0)
        
        # torch.masked_fill(a, b) fills whatever entries in a is 1 by b. So we need to invert the mask "~"".
        scores = scores.masked_fill( ~self.decoder_token_masks.repeat_interleave(beam_size, dim=0) , -1e20)
        return scores