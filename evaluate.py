import os
import json
import random
from argparse import ArgumentParser

import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
                          
from model import GenerativeModel
from config import Config
from data import IEDataset
from constants import *
from util import *
import ree_eval
import scirex_eval

# configuration
parser = ArgumentParser()
parser.add_argument('--gpu', type=int, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
args = parser.parse_args()

use_gpu = args.gpu > -1
checkpoint = torch.load(args.checkpoint, map_location=f'cuda:{args.gpu}' if use_gpu else 'cpu')
config = Config.from_dict(checkpoint['config'])

# set GPU device
config.gpu_device = args.gpu
config.use_gpu = use_gpu
# fix random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# datasets
model_name = config.bert_model_name

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=config.bert_cache_dir)
tokenizer.add_tokens(SPECIAL_TOKENS)
# special_tokens_dict = {'additional_special_tokens': SPECIAL_TOKENS}
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


# print('==============Prepare Training Set=================')
# train_set = IEDataset(config.train_file, max_length=config.max_length, gpu=use_gpu)
print('==============Prepare Dev Set=================')
dev_set = IEDataset(config.dev_file, max_length=config.max_length, gpu=use_gpu)
print('==============Prepare Test Set=================')
test_set = IEDataset(config.test_file, max_length=config.max_length, gpu=use_gpu)
vocabs = {}

# print('==============Prepare Training Set=================')
# train_set.numberize(tokenizer, vocabs)
print('==============Prepare Dev Set=================')
dev_set.numberize(tokenizer, vocabs)
print('==============Prepare Test Set=================')
test_set.numberize(tokenizer, vocabs)

if config.task == ROLE_FILLER_ENTITY_EXTRACTION:
    grit_dev = read_grit_gold_file(config.grit_dev_file)
    grit_test = read_grit_gold_file(config.grit_test_file)
elif config.task in {BINARY_RELATION_EXTRACTION, FOUR_ARY_RELATION_EXTRACTION}:
    scirex_dev = read_scirex_gold_file(config.scirex_dev_file)
    scirex_test = read_scirex_gold_file(config.scirex_test_file)


dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

output_dir = '/'.join(args.checkpoint.split('/')[:-1])
dev_result_file = os.path.join(output_dir, 'dev.out.json')
test_result_file = os.path.join(output_dir, 'test.out.json')
# initialize the model

model = GenerativeModel(config, vocabs)
model.load_bert(model_name, cache_dir=config.bert_cache_dir, tokenizer=tokenizer)

if not model_name.startswith('roberta'):
    model.bert.resize_token_embeddings(len(tokenizer))

model.load_state_dict(checkpoint['model'], strict=True)

if use_gpu:
    model.cuda(device=config.gpu_device)
epoch = 1000    
# dev set
progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                    desc='Dev {}'.format(epoch))

dev_gold_outputs, dev_pred_outputs, dev_input_tokens, dev_doc_ids, dev_documents = [], [], [], [], []

for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                        shuffle=False, collate_fn=dev_set.collate_fn):
    progress.update(1)
    outputs = model.predict(batch, tokenizer,epoch=epoch)
    decoder_inputs_outputs = generate_decoder_inputs_outputs(batch, tokenizer, model, use_gpu, config.max_position_embeddings, task=config.task)
    dev_pred_outputs.extend(outputs['decoded_ids'].tolist())
    dev_gold_outputs.extend(decoder_inputs_outputs['decoder_labels'].tolist())
    dev_input_tokens.extend(batch.input_tokens)
    dev_doc_ids.extend(batch.doc_ids)
    dev_documents.extend(batch.document)
progress.close()

dev_result = {
    'pred_outputs': dev_pred_outputs,
    'gold_outputs': dev_gold_outputs,
    'input_tokens': dev_input_tokens,
    'doc_ids': dev_doc_ids,
    'documents': dev_documents
}  
with open(dev_result_file ,'w') as f:
    f.write(json.dumps(dev_result))


if config.task == EVENT_TEMPLATE_EXTRACTION:
    dev_scores = 0
elif config.task == ROLE_FILLER_ENTITY_EXTRACTION:
    ree_preds = construct_outputs_for_ceaf(dev_pred_outputs, dev_input_tokens, dev_doc_ids, tokenizer) 
    dev_scores = ree_eval.ree_eval(ree_preds, grit_dev)
elif config.task == BINARY_RELATION_EXTRACTION:
    bre_preds = construct_outputs_for_scirex(dev_pred_outputs, dev_documents, dev_doc_ids, tokenizer, task=BINARY_RELATION_EXTRACTION)
    dev_scores = scirex_eval.scirex_eval(bre_preds, scirex_dev, cardinality=2)
elif config.task == FOUR_ARY_RELATION_EXTRACTION:
    bre_preds = construct_outputs_for_scirex(dev_pred_outputs, dev_documents, dev_doc_ids, tokenizer, task=FOUR_ARY_RELATION_EXTRACTION)
    dev_scores = scirex_eval.scirex_eval(bre_preds, scirex_dev, cardinality=4)
else:
    raise NotImplementedError
save_model = False




# test set
progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                    desc='Test {}'.format(epoch))
test_gold_outputs, test_pred_outputs, test_input_tokens, test_doc_ids, test_documents = [], [], [], [], []
test_loss = 0

for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False,
                        collate_fn=test_set.collate_fn):
    progress.update(1)
    outputs = model.predict(batch, tokenizer, epoch=epoch)
    decoder_inputs_outputs = generate_decoder_inputs_outputs(batch, tokenizer, model, use_gpu, config.max_position_embeddings, task=config.task)

    test_pred_outputs.extend(outputs['decoded_ids'].tolist())
    test_gold_outputs.extend(decoder_inputs_outputs['decoder_labels'].tolist())
    test_input_tokens.extend(batch.input_tokens)
    test_doc_ids.extend(batch.doc_ids)
    test_documents.extend(batch.document)
progress.close()


# currently use negative dev loss as validation criteria 
if config.task == EVENT_TEMPLATE_EXTRACTION:
    # TODO: call the official evaluator
    test_scores = 0
elif config.task == ROLE_FILLER_ENTITY_EXTRACTION:
    ree_preds = construct_outputs_for_ceaf(test_pred_outputs, test_input_tokens, test_doc_ids, tokenizer)
    test_scores = ree_eval.ree_eval(ree_preds, grit_test)
elif config.task == BINARY_RELATION_EXTRACTION:
    bre_preds = construct_outputs_for_scirex(test_pred_outputs, test_documents, test_doc_ids, tokenizer, task=BINARY_RELATION_EXTRACTION)
    test_scores = scirex_eval.scirex_eval(bre_preds, scirex_test, cardinality=2)
elif config.task == FOUR_ARY_RELATION_EXTRACTION:
    bre_preds = construct_outputs_for_scirex(test_pred_outputs, test_documents, test_doc_ids, tokenizer, task=FOUR_ARY_RELATION_EXTRACTION)
    test_scores = scirex_eval.scirex_eval(bre_preds, scirex_test, cardinality=4)
else:
    raise NotImplementedError

test_result = {
    'pred_outputs': test_pred_outputs,
    'gold_outputs': test_gold_outputs,
    'input_tokens': test_input_tokens,
    'doc_ids': test_doc_ids,
    'documents': test_documents
}  
with open(test_result_file,'w') as f:
    f.write(json.dumps(test_result))
        