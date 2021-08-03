'''
Copied over from https://huggingface.co/transformers/_modules/transformers/models/bart/modeling_bart.html
'''
import torch
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartModel, BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
class CopyBartForConditionalGeneration(BartForConditionalGeneration):

    def __init__(self, config: BartConfig):
        super().__init__(config)
        
        self.selected_heads = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        assert output_attentions or self.model.config.output_attentions, "output_attentions must be true"
        
        # original outputs
        outputs = self.model(input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,)
        
        if input_ids is None:
            input_ids = self._cache_input_ids

        # if self.selected_heads is None:
        # take the cross attention non-linear function
        cross_attention_non_linear = self.model.decoder.layers[-1].encoder_attn.out_proj.weight # (emb_dim, emb_dim)
        cross_attention_non_linear_sum = cross_attention_non_linear.view(self.config.decoder_attention_heads, -1).abs().sum(1) # (num_heads)
        _, selected_heads = torch.topk(cross_attention_non_linear_sum, k=self._k)
        self.selected_heads = selected_heads

        encoder_last_hidden_state = outputs.encoder_last_hidden_state # (batch, seq, hidden)
        decoder_last_hidden_state = outputs[0] #(batch, decoding_seq, hidden )
        
        
        # compute lm logits based on attention
        last_cross_attentions = outputs.cross_attentions[-1] # (batch_size, num_heads, decoding_seq_length, encoding_seq_length).
        
        
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias  #(batch_size, decoding_seq_length, emb_dim)
        
        
        
        cross_attentions_aggregate = last_cross_attentions[:,self.selected_heads,:,:].mean(dim=1) #(batch, decoding_seq_length, encoding_seq_length)

        
        dummy_input_ids = input_ids.unsqueeze(-1).expand(-1, -1, lm_logits.size(1)).transpose(1,2) # (batch, decoding_seq_length, encoding_seq_length)
        copy_logits = torch.zeros_like(lm_logits) # (batch, decoding_seq_length, emb_dim)
        copy_logits.scatter_add_(dim=2, index=dummy_input_ids, src=cross_attentions_aggregate)   
        
        
        p_gen = torch.bmm(decoder_last_hidden_state, encoder_last_hidden_state.mean(dim=1).unsqueeze(dim=-1)) # (batch, decoding_seq, 1)
        p_gen = torch.sigmoid(p_gen)


        lm_logits = F.softmax(lm_logits, dim=-1) * p_gen + copy_logits * (1 - p_gen)#(batch_size, decoding_seq_length, emb_dim)
        
        


        masked_lm_loss = None
        if labels is not None:
            # compute loss mask and fill -100 with 0
            loss_mask = labels != -100
            labels.masked_fill_(~loss_mask, 0)
            # use negative log likelihood
            gold_probs = torch.gather(lm_logits, 2, labels.unsqueeze(2)).squeeze(2)
            eps = 1e-7 # for safe log
            masked_lm_loss = - torch.log(gold_probs + eps) * self._loss_weight[labels]
            masked_lm_loss = (masked_lm_loss * loss_mask).mean()
            


        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


    

