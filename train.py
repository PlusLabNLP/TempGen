import os
import json
import time
import random
from argparse import ArgumentParser

import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, Adafactor
                          
from model import GenerativeModel
from config import Config
from data import IEDataset
from constants import *
from util import *
import ree_eval
import scirex_eval

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/generative_model.json')
args = parser.parse_args()
config = Config.from_json_file(args.config)
print(config.to_dict())

# fix random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# set GPU device
use_gpu = config.use_gpu
if use_gpu and config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# output
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_dir = os.path.join(config.log_path, timestamp)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# logger = Logger(log_dir)
output_dir = os.path.join(config.output_path, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_file = os.path.join(output_dir, 'log.txt')
with open(log_file, 'w', encoding='utf-8') as w:
    w.write(json.dumps(config.to_dict()) + '\n')
    print('Log file: {}'.format(log_file))
best_model = os.path.join(output_dir, 'best.mdl')
train_result_file = os.path.join(output_dir, 'result.train.json')
dev_result_file = os.path.join(output_dir, 'result.dev.json')
test_result_file = os.path.join(output_dir, 'result.test.json')

# datasets
model_name = config.bert_model_name

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=config.bert_cache_dir)

tokenizer.add_tokens(SPECIAL_TOKENS)
# special_tokens_dict = {'additional_special_tokens': SPECIAL_TOKENS}
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


print('==============Prepare Training Set=================')
train_set = IEDataset(config.train_file, max_length=config.max_length, gpu=use_gpu)
print('==============Prepare Dev Set=================')
dev_set = IEDataset(config.dev_file, max_length=config.max_length, gpu=use_gpu)
print('==============Prepare Test Set=================')
test_set = IEDataset(config.test_file, max_length=config.max_length, gpu=use_gpu)
vocabs = {}

print('==============Prepare Training Set=================')
train_set.numberize(tokenizer, vocabs)
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

batch_num = len(train_set) // (config.batch_size * config.accumulate_step) + \
    (len(train_set) % (config.batch_size * config.accumulate_step) != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

# initialize the model

model = GenerativeModel(config, vocabs)

model.load_bert(model_name, cache_dir=config.bert_cache_dir, tokenizer=tokenizer)

if not model_name.startswith('roberta'):
    model.bert.resize_token_embeddings(len(tokenizer))

if use_gpu:
    model.cuda(device=config.gpu_device)

# optimizer
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
        'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                   and 'crf' not in n and 'global_feature' not in n],
        'lr': config.learning_rate, 'weight_decay': config.weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                   and ('crf' in n or 'global_feature' in n)],
        'lr': config.learning_rate, 'weight_decay': 0
    }
]
if model.bert.config.name_or_path.startswith('t5'):
    optimizer = Adafactor(params=param_groups)
else:
    optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num*config.warmup_epoch,
                                           num_training_steps=batch_num*config.max_epoch)

# model state
state = dict(model=model.state_dict(),
             config=config.to_dict(),
             vocabs=vocabs)


best_dev = -np.inf
current_step = 0
best_epoch = 0
print('================Start Training================')
for epoch in range(config.max_epoch):
    
    progress = tqdm.tqdm(total=batch_num, ncols=75,
                         desc='Train {}'.format(epoch))
    optimizer.zero_grad()
    train_gold_outputs, train_pred_outputs, train_input_tokens, train_doc_ids, train_input_ids = [], [], [], [], []
    training_loss = 0
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=config.batch_size ,
            shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
        
        decoder_inputs_outputs = generate_decoder_inputs_outputs(batch, tokenizer, model, use_gpu, config.max_position_embeddings, permute_slots=config.permute_slots, task=config.task)
        decoder_input_ids = decoder_inputs_outputs['decoder_input_ids']
        
        decoder_labels = decoder_inputs_outputs['decoder_labels']
        decoder_masks = decoder_inputs_outputs['decoder_masks']
        
        loss = model(batch, decoder_input_ids, decoder_labels, tokenizer=tokenizer)['loss']
        current_step += 1
        loss = loss * (1 / config.accumulate_step)
        training_loss += loss.item()
        loss.backward()
        
        
        train_gold_outputs.extend(decoder_inputs_outputs['decoder_labels'].tolist())
        train_input_ids.extend(decoder_input_ids.tolist())
        

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    # train the last batch
    if batch_num % config.accumulate_step != 0:
        progress.update(1)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clipping)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()

    print("training loss", training_loss)
    train_result = {
        'pred_outputs': train_pred_outputs,
        'gold_outputs': train_gold_outputs,
        'input_tokens': train_input_tokens,
        'decoder_input_ids': train_input_ids,
        'doc_ids': train_doc_ids
    }  
    with open( train_result_file + f'_{epoch}','w') as f:
        f.write(json.dumps(train_result))
            
    progress.close()
    if config.max_epoch <= 50 or epoch % (config.max_epoch // 150) == 0 :
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
        with open( dev_result_file + f'_{epoch}','w') as f:
            f.write(json.dumps(dev_result))

        # TODO: call the official evaluator
        
        if config.task == ROLE_FILLER_ENTITY_EXTRACTION:
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


        if config.task == ROLE_FILLER_ENTITY_EXTRACTION:
            current_dev_score = dev_scores['micro_avg']['f1']
            save_model = current_dev_score > best_dev
        elif config.task in {BINARY_RELATION_EXTRACTION, FOUR_ARY_RELATION_EXTRACTION}:
            current_dev_score = dev_scores['f1']
            save_model = current_dev_score > best_dev
        if save_model:
            best_dev = current_dev_score
            best_epoch = epoch
            print('Saving best model')
            torch.save(state, best_model)

        
        if save_model:
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

            
            
            if config.task == ROLE_FILLER_ENTITY_EXTRACTION:
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
            with open( test_result_file + f'_{epoch}','w') as f:
                f.write(json.dumps(test_result))
                
            result = json.dumps(
                {'epoch': epoch, 'dev': dev_scores, 'test': test_scores})
            with open(log_file, 'a', encoding='utf-8') as w:
                w.write(result + '\n')
        print('Log file', log_file)
if config.task == ROLE_FILLER_ENTITY_EXTRACTION:
    get_best_score(log_file, 'micro_avg')
elif config.task in {BINARY_RELATION_EXTRACTION, FOUR_ARY_RELATION_EXTRACTION}:
    get_best_score_bre(log_file)
print(config.to_dict())