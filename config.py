import copy
import json
import os
from constants import *

from transformers import AutoConfig

class Config(object):
    def __init__(self, **kwargs):
        self.coref = kwargs.pop('coref', False)
        # bert
        self.bert_model_name = kwargs.pop('bert_model_name', 'bert-large-cased')
        self.bert_cache_dir = kwargs.pop('bert_cache_dir', None)
        self.extra_bert = kwargs.pop('extra_bert', -1)
        self.use_extra_bert = kwargs.pop('use_extra_bert', False)
        # model
        # self.multi_piece_strategy = kwargs.pop('multi_piece_strategy', 'first')
        self.bert_dropout = kwargs.pop('bert_dropout', .5)
        self.linear_dropout = kwargs.pop('linear_dropout', .4)
        self.linear_bias = kwargs.pop('linear_bias', True)
        self.linear_activation = kwargs.pop('linear_activation', 'relu')
        
        # decoding
        self.max_position_embeddings = kwargs.pop('max_position_embeddings', 2048)
        self.num_beams = kwargs.pop('num_beams', 4)
        self.decoding_method = kwargs.pop('decoding_method', "greedy")

        # files
        self.train_file = kwargs.pop('train_file', None)
        self.dev_file = kwargs.pop('dev_file', None)
        self.test_file = kwargs.pop('test_file', None)
        self.valid_pattern_path = kwargs.pop('valid_pattern_path', None)
        self.log_path = kwargs.pop('log_path', './log')
        self.output_path = kwargs.pop('output_path', './output')
        self.grit_dev_file = kwargs.pop('grit_dev_file', None)
        self.grit_test_file = kwargs.pop('grit_test_file', None)

        # training
        self.accumulate_step = kwargs.pop('accumulate_step', 1)
        self.batch_size = kwargs.pop('batch_size', 10)
        self.eval_batch_size = kwargs.pop('eval_batch_size', 5)
        self.max_epoch = kwargs.pop('max_epoch', 50)
        self.max_length = kwargs.pop('max_length', 128)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.bert_learning_rate = kwargs.pop('bert_learning_rate', 1e-5)
        self.weight_decay = kwargs.pop('weight_decay', 0.001)
        self.bert_weight_decay = kwargs.pop('bert_weight_decay', 0.00001)
        self.warmup_epoch = kwargs.pop('warmup_epoch', 5)
        self.grad_clipping = kwargs.pop('grad_clipping', 5.0)
        self.SOT_weights = kwargs.pop('SOT_weights', 100)
        self.permute_slots = kwargs.pop('permute_slots', False)
        
        self.task = kwargs.pop('task',EVENT_TEMPLATE_EXTRACTION) # task cannot be empty

        # others
        self.use_gpu = kwargs.pop('use_gpu', True)
        self.gpu_device = kwargs.pop('gpu_device', 0)
        self.seed = kwargs.pop('seed', 0)
        self.use_copy = kwargs.pop('use_copy', False)
        self.use_SAGCopy = kwargs.pop('use_SAGCopy', False)
        self.k = kwargs.pop('k', 12)
        


    @classmethod
    def from_dict(cls, dict_obj):
        """Creates a Config object from a dictionary.
        Args:
            dict_obj (Dict[str, Any]): a dict where keys are
        """
        config = cls()
        for k, v in dict_obj.items():
            setattr(config, k, v)
        return config

    @classmethod
    def from_json_file(cls, path):
        with open(path, 'r', encoding='utf-8') as r:
            return cls.from_dict(json.load(r))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def save_config(self, path):
        """Save a configuration object to a file.
        :param path (str): path to the output file or its parent directory.
        """
        if os.path.isdir(path):
            path = os.path.join(path, 'config.json')
        print('Save config to {}'.format(path))
        with open(path, 'w', encoding='utf-8') as w:
            w.write(json.dumps(self.to_dict(), indent=2,
                               sort_keys=True))
    @property
    def bert_config(self):
        
        
        return AutoConfig.from_pretrained(self.bert_model_name,
                                                    cache_dir=self.bert_cache_dir,
                                                    max_position_embeddings=self.max_position_embeddings)
        
            