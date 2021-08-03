import argparse
import json
from glob import glob
import re
from collections import defaultdict

def clean_docid(value):
    return re.sub(r'\s*\(.*$','', value)

def cleankey(key):
    return re.sub(r'[^A-Z|\s]+', '', key).strip().replace(' ','-')

ALL_KEYS = """
MESSAGE: ID
MESSAGE: TEMPLATE
INCIDENT: DATE
INCIDENT: LOCATION
INCIDENT: TYPE
INCIDENT: STAGE OF EXECUTION
INCIDENT: INSTRUMENT ID
INCIDENT: INSTRUMENT TYPE
PERP: INCIDENT CATEGORY
PERP: INDIVIDUAL ID
PERP: ORGANIZATION ID
PERP: ORGANIZATION CONFIDENCE
PHYS TGT: ID
PHYS TGT: TYPE
PHYS TGT: NUMBER
PHYS TGT: FOREIGN NATION
PHYS TGT: EFFECT OF INCIDENT
PHYS TGT: TOTAL NUMBER
HUM TGT: NAME
HUM TGT: DESCRIPTION
HUM TGT: TYPE
HUM TGT: NUMBER
HUM TGT: FOREIGN NATION
HUM TGT: EFFECT OF INCIDENT
HUM TGT: TOTAL NUMBER
""".strip().split('\n')

ALL_KEYS = set(cleankey(key) for key in ALL_KEYS)
SINGLE_VALUE_KEYS = set(cleankey(key) for key in ['MESSAGE: ID','MESSAGE: TEMPLATE','INCIDENT: DATE','INCIDENT: LOCATION','INCIDENT: TYPE','INCIDENT: STAGE OF EXECUTION','PERP: INCIDENT CATEGORY','PHYS TGT: TOTAL NUMBER'])

def gather_key_vals(chunk):
    """
    Processes the raw MUC "key file" format.  Parses one entry ("chunk").
    Returns a dictionary of key-values.
    A single key can be repeated many times.
    This function cleans up key names, but passes the values through as-is.
    
    """
    res = defaultdict(list)
    curkey = None
    for line in chunk.split('\n'):
        if line.startswith(';'):
            
            continue
        middle = 33  ## Different in dev vs test files... this is the minimum size to get all keys.
        keytext = line[:middle].strip()
        valtext = line[middle:].strip()
        if not keytext:
            ## it's a continuation
            assert curkey
        else:
            curkey = cleankey(keytext)
            assert curkey in ALL_KEYS, (curkey, line)
        
        # if it's message_id then clean value
        if curkey == cleankey('MESSAGE: ID'):
            valtext = clean_docid(valtext)
        
        
        # elif curkey == cleankey('MESSAGE: TEMPLATE') and '(OPTIONAL)' in valtext:
        #     valtext =  valtext.replace('(OPTIONAL)','').strip(' ')
            
        # do not append empty vals
        if valtext not in ['*','-']:
            if curkey in SINGLE_VALUE_KEYS:
                res[curkey] = valtext
            else:
                res[curkey].append([val.strip() for val in valtext.split('/')])
    
    return res

def combine(new_dict, all_dict):
    doc_id = new_dict[cleankey('MESSAGE: ID')]
    # remove id from message
    new_dict.pop(cleankey('MESSAGE: ID'))
    # nothing in the dictionary
    if len(new_dict) == 0:
        # put empty list
        all_dict[doc_id] = [] 
    else:
        all_dict[doc_id].append(new_dict)
    return all_dict

def parse_key_file(key_file):
    with open(key_file, 'r') as f:
        key_lines = f.readlines()
    
    key_lines = [L.strip('\n') for L in key_lines if not re.search(r'^\s*;', L)] ## comments

    data = '\n'.join(key_lines)

    # each chunk corresponds to a template annotation
    chunks = re.split(r'\n\n+|\n(?=0\. )', data)
    chunks = [c.strip() for c in chunks if c.strip()]

    # key by doc_id
    all_chunk_dict = defaultdict(list)

    
    for chunk in chunks:
        keyvals = gather_key_vals(chunk)
        all_chunk_dict = combine(keyvals, all_chunk_dict)
    
    return all_chunk_dict

def add_documents(all_key_file_dict, input_corpus):
    '''
    Attach document into each annotation.
    '''

    res = {}

    with open(input_corpus, 'r') as f:
        documents = [json.loads(l) for l in f.readlines()]

    for doc_id, annotation in all_key_file_dict.items():

        input_document = [ '\n'.join([document['dateline']]+ document['tags'] + [document['text']]) for document in documents if document['docid'] == doc_id][0]

        res[doc_id] = {
            'document': input_document,
            'annotation': annotation
        }

    return res

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_corpus', type=str, help="corpus containing MUC documents represented in a list of json")
    p.add_argument('--input_pattern', type=str, help="input patters that will be passed to glob to fetch file names.") 
    p.add_argument('--output_path',type=str, help="path to store the output json file.")
    args = p.parse_args()

    key_files = glob(f"{args.input_pattern}*")
    all_key_file_dict = {}
    for key_file in key_files:
        one_key_file_dict = parse_key_file(key_file)
        all_key_file_dict.update(one_key_file_dict)
    # attach document
    all_key_file_dict = add_documents(all_key_file_dict, args.input_corpus)

    with open(args.output_path, 'w') as f:
        f.write(json.dumps(all_key_file_dict))