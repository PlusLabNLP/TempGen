from typing import Dict, Any, List
import argparse
import json

from itertools import combinations

# copied over from https://github.com/allenai/SciREX/blob/master/scirex_utilities/entity_utils.py
available_entity_types_sciERC = ["Material", "Metric", "Task", "Generic", "OtherScientificTerm", "Method"]
map_available_entity_to_true = {"Material": "dataset", "Metric": "metric", "Task": "task", "Method": "model_name"}
map_true_entity_to_available = {v: k for k, v in map_available_entity_to_true.items()}
used_entities = list(map_available_entity_to_true.keys())

def generate_relations(doc: Dict[str, Any], cardinality) -> List[Dict]:
    '''
    Break down 4-ary relations into binary relations.
    '''    
    def get_mentions(clusters, doc_tokens, entity_name):
        res = []
        cluster = clusters[entity_name] 
        # TODO: set does not preserve order so currently I use for loop to keep the order of mentions while removing duplicates.
        for start_tok, end_tok in cluster:
            mention = ' '.join(doc_tokens[start_tok:end_tok])
            if mention not in res:
                res.append(mention)
        return res

    res = []
    for types in combinations(used_entities, cardinality):
        relations = [tuple((t, x[t]) for t in types) for x in doc['n_ary_relations']]            

        # make sure each entity has at least one cluster and make (entity_1, entity_2, relation) unique
        relations = set([x for x in relations if has_all_mentions(doc, x)])
        
        
        for relation in relations:
            current_relation_dict = {}
            for entity in relation:
                entity_type, entity_name = entity
                entity_mentions = get_mentions(doc['coref'], doc['words'], entity_name)    
                current_relation_dict[entity_type] = [entity_mentions] # we need to make it a list of list to comply with the convention in data.py 

            res.append(current_relation_dict)
    return res


def has_all_mentions(doc: Dict[str, Any], relation):
    '''
    
    Make sure each entity has at least one mention.
    '''
    has_mentions = all(len(doc["coref"][x[1]]) > 0 for x in relation)
    return has_mentions


def tokens_to_string(tokens):
    return ' '.join(tokens)

def process_document(doc: Dict[str, Any], cardinality: int) -> Dict[str, Any]:
    
    assert cardinality in {2, 4}, "Only support binary and 4-ary relations"

    relations = generate_relations(doc, cardinality)
    
    

    doctext = tokens_to_string(doc['words'])
    return {
        'doc_id': doc['doc_id'],        
        'document': doctext,
        'annotation': relations,
        
    }

def process_file(input_path: str, output_path: str, cardinality:int):

    with open(input_path, 'r') as f:
        input_data = [json.loads(l) for l in f.readlines()]
    
    processed_docs = {}
    for input_doc in input_data:
        processed_data = process_document(input_doc, cardinality)
        doc_id = processed_data.pop('doc_id')
        
        processed_docs[doc_id] = processed_data
    
    # store in json format    
    with open(output_path, 'w') as f:
        json.dump(processed_docs, f)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    
    p.add_argument('--input_path', type=str) 
    p.add_argument('--output_path',type=str)
    p.add_argument('--cardinality',type=int)
    args = p.parse_args()

    process_file(args.input_path, args.output_path, args.cardinality)    