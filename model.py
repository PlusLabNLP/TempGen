
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoModel, BartForConditionalGeneration, EncoderDecoderModel, RobertaConfig, EncoderDecoderConfig, AutoModelForCausalLM
from transformers import BeamSearchScorer, LogitsProcessorList
from logits_processor import MaskNonInputLogitsProcessor
from collections import Counter
import copy
from torch.nn.modules import TransformerDecoderLayer, TransformerDecoder, LayerNorm
from tqdm import tqdm
from copy_bart import CopyBartForConditionalGeneration
from sagcopy import SAGCopyBartForConditionalGeneration
from constants import *


        
class GenerativeModel(nn.Module):
    def __init__(self,
                 config,
                 vocabs):
        super().__init__()
        

        # vocabularies
        self.vocabs = vocabs
        

        # BERT encoder
        bert_config = config.bert_config
        bert_config.output_hidden_states = True
        self.bert_dim = bert_config.hidden_size
        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.bert_dim *= 2
        self.bert_config = bert_config
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.max_position_embeddings = config.max_position_embeddings
        self.num_beams = config.num_beams
        self.decoding_method = config.decoding_method
        self.SOT_weights = config.SOT_weights
        self.max_length = config.max_length
        self.use_copy = config.use_copy
        self.use_SAGCopy = config.use_SAGCopy
        self._k = config.k
        # TODO: may need to tune weight for padding token
        
        # self.decoder_criteria = torch.nn.CrossEntropyLoss()
    
    def load_bert(self, name, cache_dir=None, tokenizer=None):
        """Load the pre-trained BERT model (used in training phrase)
        :param name (str): pre-trained BERT model name
        :param cache_dir (str): path to the BERT cache directory
        """
        print('Loading pre-trained BERT model {}'.format(name))
        

        if self.use_copy:
            self.bert = CopyBartForConditionalGeneration.from_pretrained(name, cache_dir=cache_dir, output_attentions=True)
            self.bert._k = self._k
        elif self.use_SAGCopy:
            self.bert = SAGCopyBartForConditionalGeneration.from_pretrained(name, cache_dir=cache_dir, output_attentions=True, output_hidden_states=True)
        elif name.startswith('roberta'):
            encoder = AutoModel.from_pretrained(name, cache_dir=cache_dir, output_hidden_states=True)
            decoder = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir, output_hidden_states=True, is_decoder=True, add_cross_attention=True)
            encoder.resize_token_embeddings(len(tokenizer))
            decoder.resize_token_embeddings(len(tokenizer))
            
            
            
            self.bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
            self.bert.config.tie_encoder_decoder=True
            self.bert.tie_weights()
            self.bert.config.decoder_start_token_id = tokenizer.eos_token_id
            
        else:
            self.bert = AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir=cache_dir) 
            # if self.max_length > 1024:
            #     # https://github.com/pytorch/fairseq/issues/1685#issuecomment-621520129
            #     # adapted from https://github.com/huggingface/transformers/issues/4277#issuecomment-629452698
            #     sd = self.bert.state_dict()
            #     shorter_pos_embeds = sd['model.encoder.embed_positions.weight']
            #     new_config = self.bert.config
            #     new_config.max_position_embeddings = self.max_length
            #     new_model = BartForConditionalGeneration(new_config)
            #     correctly_shaped_pos_weight = new_model.model.encoder.embed_positions.weight.cuda()
            #     correctly_shaped_pos_weight[:shorter_pos_embeds.shape[0]] = shorter_pos_embeds.cuda()
            #     # starting from 2 because BART reserve 2 speical tokens for <s> and </s> it seems
            #     correctly_shaped_pos_weight[shorter_pos_embeds.shape[0]:] = shorter_pos_embeds[2:self.max_length - shorter_pos_embeds.shape[0] + 4].cuda()
            #     sd['model.decoder.embed_positions.weight'] = correctly_shaped_pos_weight
            #     sd['model.encoder.embed_positions.weight'] = correctly_shaped_pos_weight
            #     new_model.load_state_dict(sd, strict=True)
            #     self.bert = new_model
                


        
    def forward(self, batch, decoder_input_ids=None, decoder_labels=None, decoder_masks=None, logger=None, tag=None, step=None, tokenizer=None):
        
        res = {}
        # increase weight for <SOT> 
        vocab_size = len(tokenizer)
        
        weight = torch.ones(vocab_size).to(batch.input_ids.device)
        self.bert._loss_weight = weight
        self.bert._vocab_size = vocab_size
        
        if self.use_copy or self.use_SAGCopy:
            bart_outputs = self.encode(batch, decoder_input_ids=decoder_input_ids, decoder_labels=decoder_labels)
        else:
            bart_outputs = self.encode(batch, decoder_input_ids=decoder_input_ids)
        
        # if labels provided, assign loss
        if decoder_labels is not None:
            
            if self.use_copy or self.use_SAGCopy:
                weight[tokenizer.convert_tokens_to_ids(START_OF_TEMPLATE)] = self.SOT_weights
                loss = bart_outputs.loss
            else:
                weight[tokenizer.convert_tokens_to_ids(START_OF_TEMPLATE)] = self.SOT_weights
                # weight[tokenizer.eos_token_id] = 0.05
                loss = torch.nn.functional.cross_entropy(input=bart_outputs.logits.view(-1, vocab_size), target=decoder_labels.view(-1), weight=weight)

            res['loss'] = loss

        return res
        
    def encode(self, batch, decoder_input_ids=None, decoder_labels=None, decoder_masks=None):
        '''
        Encode the input documents
        '''
        
        return self.bert(input_ids=batch.input_ids,
                            attention_mask=batch.attention_masks, #1 for tokens that are not masked, 0 for tokens that are masked.
                            decoder_input_ids=decoder_input_ids, # For translation and summarization training, decoder_input_ids should be provided. If no decoder_input_ids is provided, the model will create this tensor by shifting the input_ids to the right for denoising pre-training following the paper.
                            labels=decoder_labels, 
                            # decoder_attention_mask=decoder_masks, #Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
                            return_dict=True,
                            output_hidden_states=True,
                            
                            )
    
    def beam_search(self, batch, num_beams, decoding_length, decoder_token_masks=None):
        '''
        Adapted from https://huggingface.co/transformers/main_classes/model.html?highlight=beamsearchscorer
        Do stardard beam search
        '''
        beam_scorer = BeamSearchScorer(
            batch_size=batch.input_ids.size(0),
            max_length=decoding_length,
            num_beams=num_beams,
            device=self.bert.device,
        )
        
        

        logits_processor = LogitsProcessorList([
        #    MaskNonInputLogitsProcessor(decoder_token_masks),
        ])

        # logits_warper = LogitsProcessorList([
        #     TopKLogitsWarper(50),
            # TemperatureLogitsWarper(0.7),
        # ])
        
        # seems that this is required if our model is a encoder-decoder architecture.
        model_kwargs = {
            "encoder_outputs": self.bert.get_encoder()(batch.input_ids.repeat_interleave(num_beams, dim=0), batch.attention_masks.repeat_interleave(num_beams, dim=0), return_dict=True),
        }
        # huggingface beamsearch workaround
        self.bert._cache_input_ids = batch.input_ids

        # create token for start decoding.
        decoder_input_ids = torch.ones((num_beams * batch.input_ids.size(0), 1), device=self.bert.device, dtype=torch.long)
        decoder_input_ids = decoder_input_ids * self.bert.config.decoder_start_token_id
        
        decoded_ids = self.bert.beam_search(decoder_input_ids, beam_scorer, max_length=decoding_length, logits_processor=logits_processor, **model_kwargs)
        
        return decoded_ids


    def predict(self, batch, tokenizer, epoch=None):
        self.eval()
        
        

        with torch.no_grad():
            
            decoding_length = self.max_position_embeddings-1
            # when epoch < 4, the model generates trash
            if epoch is not None and epoch < 10:
                decoding_length = 10

            
            # only those token present in the input document and the special tokens can be decoded.
            # (batch, num_tokens)
            decoder_token_masks = torch.zeros(batch.input_ids.size(0), len(tokenizer) ,device=batch.input_ids.device, dtype=torch.bool)

            for batch_idx, input_ids in enumerate(batch.input_ids):    
                decoder_token_masks[batch_idx, input_ids] = 1

            # TODO: these can be cached in the __init__ function so we don't need to do it repeatedly.
            decoder_token_masks[:, tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)] = 1
            decoder_token_masks[:, tokenizer.eos_token_id] = 1
            decoder_token_masks[:, tokenizer.bos_token_id] = 1
            for role in REE_ROLES:
                decoder_token_masks[:, tokenizer.encode(role, add_special_tokens=False)]  = 1
            
            if self.decoding_method == 'greedy':
                # Adapted part of the code from https://huggingface.co/blog/encoder-decoder
                decoded_ids = torch.LongTensor([[self.bert.config.decoder_start_token_id] * len(batch.input_ids)]).to(batch.input_ids.device).reshape(-1,1)
                
                # pass input_ids to encoder and to decoder and pass BOS token to decoder to retrieve first logit
                bart_outputs = self.bert(batch.input_ids, attention_mask=batch.attention_masks, decoder_input_ids=decoded_ids, use_cache=True, return_dict=True)

                # encode encoder input_ids once
                encoded_sequence = (bart_outputs.encoder_last_hidden_state,)    
                
                
                # get next token id and append it to decoded list
                lm_logits = bart_outputs.logits 
                next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
                decoded_ids = torch.cat([decoded_ids, next_decoder_input_ids], axis=-1)

                # use past_key_values to speed up decoding
                past_key_values = bart_outputs.past_key_values
                
                
                # only those token present in the input document and the special tokens can be decoded.
                    
                for i in range(decoding_length):

                    bart_outputs = self.bert(batch.input_ids, encoder_outputs=encoded_sequence, past_key_values=past_key_values, decoder_input_ids=next_decoder_input_ids, use_cache=True, return_dict=True)
                    lm_logits = bart_outputs.logits 

                    # TODO: this is incorrect, will implement in the future if necessary
                    # lm_logits[:,-1] = lm_logits[:,-1] * decoder_token_masks
                    past_key_values = bart_outputs.past_key_values

                    # sample last token with highest prob again
                    next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
                    # concat again
                    decoded_ids = torch.cat([decoded_ids, next_decoder_input_ids], axis=-1)
                    
                    if torch.all(next_decoder_input_ids == tokenizer.eos_token_id):
                        break
                # decoded_ids = self.bert.generate(input_ids=batch.input_ids, attention_mask=batch.attention_masks, max_length=100)
            elif self.decoding_method == "beam_search":
                decoded_ids = self.beam_search(batch, num_beams=4, decoding_length=decoding_length, decoder_token_masks=decoder_token_masks)
            else:
                raise NotImplementedError
            res = {
                'decoded_ids':decoded_ids
            }
            

        self.train()
        return res
    
