from torch.utils.data import Dataset
from constants import *
from collections import namedtuple
from util import token2sub_tokens
import json
import torch

instance_fields = [
    'doc_id', 'input_ids', 'attention_mask','decoder_input_chunks', 'input_tokens','document'
]

batch_fields = [
    'doc_ids', 'input_ids', 'attention_masks','decoder_input_chunks', 'input_tokens','document'
]

Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))

class IEDataset(Dataset):
    def __init__(self, path, max_length=128, gpu=False):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.path = path
        self.data = []
        self.max_length = max_length
        self.gpu = gpu
        
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    
    def load_data(self):
        """Load data from file."""
        overlength_num = title_num = 0
        with open(self.path, 'r', encoding='utf-8') as r:
            
            self.data = json.loads(r.read())


    def create_decoder_input_chunks(self, templates, tokenizer):
        
        '''
        `templates` is a list of dict.
         [
            {
                MESSAGE-TEMPLATE': '1',
                'INCIDENT-DATE': '28 AUG 89',
                ...
            },
            {
                'MESSAGE-TEMPLATE': '2',
                'INCIDENT-DATE': '- 30 AUG 89',
                ...
            }
        ]
        Parse the templates and create a chunk of ids
        [tokenizer.eos_token_id, [[template_1_entity_1],[template_1_entity_2], ...],[[template_2_entit_1],[template_2_entity_2],...], tokenizer.sep_token_id ]
        '''
       

        # Bart uses the eos_token_id as the starting token for decoder_input_ids generation. If past_key_values is used, optionally only the last decoder_input_ids have to be input (see past_key_values)
        res = []
        for template in templates:
            current_template_chunk = []
            for entity_key, entity_values in template.items():
                

                if isinstance(entity_values, list):
                    for entity_value in entity_values:
                        
                        # Add " " so that the token will be the same subtoken as the input document
                        mentions = [[START_OF_ENTITY,  " " + mention.strip(" ") +" ", END_OF_ENTITY] for mention in entity_value ]
                        entity = []

                        # create a chunk for 
                        for mention in mentions:
                            
                            entity_tokens = [START_OF_SLOT_NAME, entity_key, END_OF_SLOT_NAME] + mention

                            mention_chunk = []
                            for entity_token in entity_tokens:
                                mention_chunk += token2sub_tokens(tokenizer, entity_token)
                            entity.append(mention_chunk)
                        current_template_chunk.append(entity)
                else:
                    raise NotImplementedError
            
            res.append(current_template_chunk)
        
        
        return res


    def numberize(self, tokenizer, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """
        

        data = []
        for doc_id, content in self.data.items():
            
            document = content['document']
            annotation = content['annotation']

            
            input_ids = tokenizer([document], max_length=self.max_length, truncation=True)['input_ids'][0]

            pad_num = self.max_length - len(input_ids)
            attn_mask = [1] * len(input_ids) + [0] * pad_num    
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_num

            
            decoder_input_chunks = self.create_decoder_input_chunks(annotation, tokenizer)
            
            
            assert len(input_ids) == self.max_length, len(input_ids)
            
            input_tokens = tokenizer.decode(input_ids)
            # print("decoder_input_chunks", decoder_input_chunks)
            instance = Instance(
                doc_id=doc_id,
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_input_chunks=decoder_input_chunks,
                input_tokens=input_tokens,
                document=document
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_input_ids = []
        batch_attention_masks = []
        batch_decoder_input_chunks = []
        batch_input_tokens = []
        batch_document = []

        doc_ids = [inst.doc_id for inst in batch]
        
        for inst in batch:
            batch_input_ids.append(inst.input_ids)
            batch_attention_masks.append(inst.attention_mask)
            batch_decoder_input_chunks.append(inst.decoder_input_chunks)
            batch_input_tokens.append(inst.input_tokens)
            batch_document.append(inst.document)
        
        if self.gpu:
            batch_input_ids = torch.cuda.LongTensor(batch_input_ids)
            batch_attention_masks = torch.cuda.FloatTensor(batch_attention_masks)

        else:
            batch_input_ids = torch.LongTensor(batch_input_ids)
            batch_attention_masks = torch.FloatTensor(batch_attention_masks)
        
        return Batch(
            doc_ids=doc_ids,
            input_ids=batch_input_ids,
            attention_masks=batch_attention_masks,
            decoder_input_chunks=batch_decoder_input_chunks,
            input_tokens=batch_input_tokens,
            document=batch_document
        )