import torch
import numpy as np
import random 
from collections import OrderedDict
import json
from tabulate import tabulate
from typing import Dict, List, Tuple
from constants import *

import re



class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)


def token2sub_tokens(tokenizer, token):
    """
    Take in a string value and use tokenizer to tokenize it into subtokens.
    Return a list of sub tokens.
    """
    res = []
    for sub_token in tokenizer.tokenize(token):
        # make sure it's not an empty string
        if len(sub_token) > 0: 
            res.append(tokenizer.convert_tokens_to_ids(sub_token))
    return res
# def generate_decoder_inputs_outputs(batch, tokenizer, model, use_gpu, max_position_embeddings, permute_slots=False, task=None):

#     if task == ROLE_FILLER_ENTITY_EXTRACTION:
#         return generate_decoder_inputs_outputs_ree(batch, tokenizer, model, use_gpu, max_position_embeddings, permute_slots)
#     elif task == BINARY_RELATION_EXTRACTION:
#         return generate_decoder_inputs_outputs_bre(batch, tokenizer, model, use_gpu, max_position_embeddings, permute_slots)
#     else:
#         raise NotImplementedError

def format_inputs_outputs(flattened_seqs, tokenizer, use_gpu, max_position_embeddings):
    
    max_seq_len = max([len(seq) for seq in flattened_seqs]) 

    # cannot be greater than position embeddings
    max_seq_len = min(max_position_embeddings, max_seq_len)    

    # create padding & mask
    decoder_input_ids = []
    decoder_masks = []
    decoder_labels = []

    
    for flattened_seq in flattened_seqs:
        
        # minus 1 because mask should match the length of input_ids
        mask = [1] * len(flattened_seq) + [0] * (max_seq_len - len(flattened_seq)-1)

        # padding. 
        flattened_seq += [tokenizer.pad_token_id] * (max_seq_len - len(flattened_seq))
        # flattened_seq += [tokenizer.pad_token_id] * (max_seq_len - len(flattened_seq))
        
        # make sure they do not exceeed max_seq_len -1
        mask = mask[:max_seq_len-1]
        flattened_seq = flattened_seq[:max_seq_len]

        input_ids = flattened_seq[:-1]
        labels = flattened_seq[1:]

        # For some reason, it seems huggingface use -100 to denote tokens that we don't want to compute loss on.
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

        decoder_input_ids.append(input_ids)
        decoder_labels.append(labels)
        decoder_masks.append(mask)
    
    
    
    # form tensor
    if use_gpu:
        decoder_input_ids = torch.cuda.LongTensor(decoder_input_ids)
        decoder_labels = torch.cuda.LongTensor(decoder_labels)
        decoder_masks = torch.cuda.FloatTensor(decoder_masks)

    else:
        decoder_input_ids = torch.LongTensor(decoder_input_ids)
        decoder_labels = torch.LongTensor(decoder_labels)
        decoder_masks = torch.FloatTensor(decoder_masks)
    
    
    res = {
        'decoder_input_ids': decoder_input_ids,
        'decoder_labels': decoder_labels,
        'decoder_masks': decoder_masks
    }
    return res



def generate_decoder_inputs_outputs(batch, tokenizer, model, use_gpu, max_position_embeddings, permute_slots=False, task=ROLE_FILLER_ENTITY_EXTRACTION):
    '''
    Process decoder_input_chunks and produce a dictionary with keys decoder_input_ids and decoder_labels.
    decoder_input_chunks is a list where each element correspond to annotation of a document.
    '''
    decoder_input_chunks = batch.decoder_input_chunks

    flattened_seqs = []

    for decoder_input_chunk in decoder_input_chunks:
        '''
        decoder_input_chunk: [[[template_1_entity_1],[template_1_entity_2], ..., ],[, [template_2_entit_1],[template_2_entity_2]] ]
        '''
        
        flatten_entities = []
        # shuffle templates

        for template in decoder_input_chunk:
            # shuffle the slots in each template.
            if permute_slots:
                template = template.copy()
                random.shuffle(template)
            
            # if BRE, we need to determine which mention to take beforehand 
            if task in {BINARY_RELATION_EXTRACTION, FOUR_ARY_RELATION_EXTRACTION}:
                # assuming each entity has different first meniton, we use this to construct a map that determines
                # which mention to sample
                # TODO: ensure that the selected mention appear in the input document (not cut off)
                first_mention2mention_idx :Dict[Tuple, int] = {}
                for entity in template:
                    first_mention2mention_idx[tuple(entity[0])] = random.randint(0, len(entity)-1) # randint includes the boundaries on both side.

            flatten_entities.append(tokenizer.convert_tokens_to_ids(START_OF_TEMPLATE))
            for entity in template:
                if task == ROLE_FILLER_ENTITY_EXTRACTION:
                    mention_chunk = random.choice(entity)
                elif task in {BINARY_RELATION_EXTRACTION, FOUR_ARY_RELATION_EXTRACTION}:
                    mention_idx = first_mention2mention_idx[tuple(entity[0])]
                    mention_chunk = entity[mention_idx]
                else:
                    raise NotImplementedError

                for sub_token in mention_chunk:
                    flatten_entities.append(sub_token)
            # <EOT>
            flatten_entities.append(tokenizer.convert_tokens_to_ids(END_OF_TEMPLATE))
        '''
        flattened_seq should looks like [tokenizer.eos_token_id, tokenizer.bos_token_id, <SOT>, <SOSN>, slot, name, <EOSN>, <SOE>, entity, <EOE>,, ..., <EOT>, tokenizer.eos_token_id]
        '''
        if model.bert.config.name_or_path.startswith('facebook/bart') or model.bert.config.name_or_path.startswith('sshleifer/distilbart'):
            flattened_seq = [model.bert.config.decoder_start_token_id, tokenizer.bos_token_id] + flatten_entities + [tokenizer.eos_token_id]
        elif model.bert.config.name_or_path.startswith('t5') or model.bert.config.name_or_path.startswith('google/pegasus') :
            # t5 does not have <s> in the decoded string
            flattened_seq = [model.bert.config.decoder_start_token_id] + flatten_entities + [tokenizer.eos_token_id]
        elif model.bert.config.decoder._name_or_path.startswith('roberta'):
            flattened_seq = [model.bert.config.decoder_start_token_id] + flatten_entities + [tokenizer.eos_token_id]
        else:
            print("model name ", model.bert.config)
            raise NotImplementedError
        
        
        # flattened_seq = flattened_seq
        flattened_seqs.append(flattened_seq)


    res = format_inputs_outputs(flattened_seqs, tokenizer, use_gpu, max_position_embeddings)

    return res

def construct_outputs_for_scirex(preds, input_documents, doc_ids, tokenizer, task):
    res = dict()

    if task == BINARY_RELATION_EXTRACTION:
        cardinality = 2
    elif task == FOUR_ARY_RELATION_EXTRACTION:
        cardinality = 4
    else:
        raise NotImplementedError

    for predicted_id_sequence, input_document, doc_id in zip(preds, input_documents, doc_ids):
        # convert id to tokens
        predicted_sequence = tokenizer.decode(predicted_id_sequence)
        res[doc_id] = extract_relations_from_sequence(predicted_sequence, input_document, cardinality)

    return res


def extract_relations_from_sequence(predicted_sequence: str, input_document: str, cardinality: int = 2):
    
    predicted_relations : List[Dict[str, str]] = []
    
    # remove the first </s>
    predicted_sequence = predicted_sequence[4:]

    # we should not decode beyond the second </s>
    try:
        first_eos_index = predicted_sequence.index('</s>')
        predicted_sequence = predicted_sequence[:first_eos_index]
    except:
        pass

    predicted_relation_sequences = predicted_sequence.replace('<SOT>','').replace('<s>','').split('<EOT>')
    
    for seq in predicted_relation_sequences:
        
        entity_types = re.findall('<SOSN>(.*?(?=<EOSN>))',seq)
        entity_types = [et.strip() for et in entity_types]
        entity_names = [entity_name.strip() for entity_name in re.findall('<SOE>(.*?(?=<EOE>))',seq)]
        entity_names = [en.strip() for en in entity_names]

        if len(entity_types) == len(entity_names) == cardinality and \
            dict(zip(entity_types, entity_names)) not in predicted_relations:
                predicted_relations.append(dict(zip(entity_types, entity_names)))
            
    
    return predicted_relations

def construct_outputs_for_ceaf(preds, input_documents, doc_ids, tokenizer):
    '''

    input_documents: a list of decoded document (str)
    
    '''
    res = OrderedDict()
    for predicted_id_sequence, input_document, doc_id in zip(preds, input_documents, doc_ids):

        # convert id to tokens
        predicted_sequence = tokenizer.decode(predicted_id_sequence)
        
        # for unknown reason GRIT do this processing for docid
        doc_id = docid = str(int(doc_id.split("-")[0][-1])*10000 + int(doc_id.split("-")[-1]))

        # transform into doc 
        res[doc_id] = event_templates_to_ceaf(predicted_sequence, input_document)

    return res
    

def event_templates_to_ceaf(event_template_sequence: str, input_document: str):
    '''
    Turns a sequence of event templates into a dictionary
    e.g.
    </s><s><SOT><SOSN>PerpInd<EOSN><SOE>salvadoran rightist sectors<EOE><SOSN>PerpInd<EOSN><SOE>soldiers<EOE><SOSN>Victim<EOSN><SOE>hector oqueli colindres<EOE><SOSN>Victim<EOSN><SOE>hilda flores<EOE><EOT></s>
    -> {
        'PerpInd':[
            [
                ["salvadoran rightist sectors"],
                
            ],
            [
                ["soldiers"]
            ]
        ],
        'Victim':[
            [
                ['hector oqueli colindres'],
            ]
            [
                ['hilda flores']
            ]
        ]
        
    }
    '''

    # remove the first </s>
    event_template_sequence = event_template_sequence[4:]

    # we should not decode beyond the second </s>
    try:
        first_eos_index = event_template_sequence.index('</s>')
        event_template_sequence = event_template_sequence[:first_eos_index]
    except:
        pass
    res = {
        'PerpInd':[],
        'PerpOrg':[],
        'Target':[],
        'Victim':[],
        'Weapon':[]
    }
    prev_slot_name = None
    prev_tag = None # this is for determining whether a mention is in the same entity cluster as the previous mention
    try:
        while event_template_sequence:
            # print(event_template_sequence)
            # print(res)
            # if encountered these, skip
            if event_template_sequence.startswith(START_OF_TEMPLATE):
                event_template_sequence = event_template_sequence[len(START_OF_TEMPLATE):]
                continue
            elif event_template_sequence.startswith('<s>'):
                event_template_sequence = event_template_sequence[len('<s>'):]
                continue
            
            elif event_template_sequence.startswith(START_OF_SLOT_NAME):
                if END_OF_SLOT_NAME in event_template_sequence:
                    end_of_slot_name_index = event_template_sequence.index(END_OF_SLOT_NAME)
                    current_slot_name = event_template_sequence[len(START_OF_SLOT_NAME):end_of_slot_name_index]
                    slot_name_length = len(current_slot_name)

                    event_template_sequence = event_template_sequence[len(START_OF_SLOT_NAME)+len(END_OF_SLOT_NAME)+slot_name_length:]
                    
                    current_slot_name = current_slot_name.strip()
                    # if the current solt name is not valid, set it to None
                    if current_slot_name not in res.keys():
                        current_slot_name = None
                        continue

                    prev_tag = SLOT_NAME_TAG
                    new_slot_name_set = True
                    
                    
                
                else:
                    # if <SOSN> tag is not ending with a <EOSN> tag, the sequence is problematic, end decoding
                    break
                
            elif event_template_sequence.startswith(START_OF_ENTITY):
                if END_OF_ENTITY in event_template_sequence:
                    end_of_entity_index = event_template_sequence.index(END_OF_ENTITY)
                    mention = event_template_sequence[len(START_OF_ENTITY): end_of_entity_index].strip()
                    mention_length = len(mention)
                    event_template_sequence = event_template_sequence[len(START_OF_ENTITY)+len(END_OF_ENTITY) +mention_length :]
                
                else: 
                    # grab whatever we have left in the sequence and append it to the current result.
                    mention = event_template_sequence[len(START_OF_ENTITY): ]
                    event_template_sequence = ''

                # the extracted mention string must be part of the input document for the role-filler entity extraction task
                if mention in input_document:
                    
                    # if previous tag is entity, this means the current mention and the previous mention belongs to the same entity cluster
                    if prev_tag == ENTITY_TAG:
                        # append the current mention to the last entity cluster
                        res[current_slot_name][-1].append(mention)
                    else:
                        # append a new cluster
                        res[current_slot_name].append([mention])


                
                prev_tag = ENTITY_TAG


            else:
                # if nothing match, reduce the sequence length by 1 and move forward
                event_template_sequence = event_template_sequence[1:]

    except Exception as e:
        
        print(event_template_sequence)

    return res


def beam_search():
    """
    Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/generation_utils.py#L985
    """
    
    pass

def read_grit_gold_file(file: str):
    golds = OrderedDict()
    with open(file, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                docid = str(int(line["docid"].split("-")[0][-1])*10000 + int(line["docid"].split("-")[-1]))

                extracts_raw = line["extracts"]

                extracts = OrderedDict()
                for role, entitys_raw in extracts_raw.items():
                    extracts[role] = []
                    for entity_raw in entitys_raw:
                        entity = []
                        for mention_offset_pair in entity_raw:
                            entity.append(mention_offset_pair[0])
                        if entity:
                            extracts[role].append(entity)
                golds[docid] = extracts
    return golds

def read_scirex_gold_file(file: str) :
    return [json.loads(line) for line in open(file)]

def construct_table(result):
    def format_string(score):
        return f'{score*100:.2f}'

    table = [["role", "prec", "rec",'f1']]
    for key, values in result.items():
        table.append( [key, format_string(values['p']), format_string(values['r']), format_string(values['f1']) ])
    
    return tabulate(table, headers="firstrow", tablefmt="grid")

def get_best_score(log_file: str, role: str):

     with open(log_file, 'r', encoding='utf-8') as r:
        config = r.readline()

        best_scores = []
        best_dev_score = 0
        for line in r:
            record = json.loads(line)
            dev = record['dev']
            test = record['test']
            epoch = record['epoch']
            
            if dev[role]['f1'] > best_dev_score:
                best_dev_score = dev[role]['f1']
                best_scores = [dev, test, epoch]

        print('Best Epoch: {}'.format(best_scores[-1]))
        
        best_dev, best_test, epoch = best_scores
        print("Dev")
        print(construct_table(best_dev))
        print("Test")
        print(construct_table(best_test))

def get_best_score_bre(log_file: str):

     with open(log_file, 'r', encoding='utf-8') as r:
        config = r.readline()

        best_scores = []
        best_dev_score = 0
        for line in r:
            record = json.loads(line)
            dev = record['dev']
            test = record['test']
            epoch = record['epoch']
            
            if dev['f1'] > best_dev_score:
                best_dev_score = dev['f1']
                best_scores = [dev, test, epoch]

        print('Best Epoch: {}'.format(best_scores[-1]))
        
        best_dev, best_test, epoch = best_scores
        print("Dev")
        print(best_dev)
        print("Test")
        print(best_test)