import argparse
import json
import nltk
# these are for splitting doctext to sentences 
nltk.download('punkt')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def process_entities(entities):

    '''
    [   
        [
            ['guerrillas', 37], 
            ['guerrilla column', 349]
        ],
        [
            ['apple', 45]
        ],
        [
            ['banana', 60]
        ]
    ]
    -> [['guerrillas, guerrilla column'], ['apple'], ['banana']]
    '''

    res = []
    for entity in entities:

        # take only the string 
        res.append([mention[0] for mention in entity])     

    return res

def convert(doc, capitalize=False):
    '''
    doc: a dictionary that has the following format:

    {'docid': 'TST1-MUC3-0001',
    'doctext': 'the guatemala army denied today that guerrillas attacked the "santo tomas" presidential farm, located on the pacific side, where president cerezo has been staying since 2 february.    a report published by the "cerigua" news agency -- mouthpiece of the guatemalan national revolutionary unity (urng) -- whose main offices are in mexico, says that a guerrilla column attacked the farm 2 days ago.    however, armed forces spokesman colonel luis arturo isaacs said that the attack, which resulted in the death of a civilian who was passing by at the time of the skirmish, was not against the farm, and that president cerezo is safe and sound.    he added that on 3 february president cerezo met with the diplomatic corps accredited in guatemala.    the government also issued a communique describing the rebel report as "false and incorrect," and stressing that the president was never in danger.    col isaacs said that the guerrillas attacked the "la eminencia" farm located near the "santo tomas" farm, where they burned the facilities and stole food.    a military patrol clashed with a rebel column and inflicted three casualties, which were taken away by the guerrillas who fled to the mountains, isaacs noted.    he also reported that guerrillas killed a peasant in the city of flores, in the northern el peten department, and burned a tank truck.',
    'extracts': {'PerpInd': [[['guerrillas', 37], ['guerrilla column', 349]]],
    'PerpOrg': [[['guatemalan national revolutionary unity', 253],
        ['urng', 294]]],
    'Target': [[['"santo tomas" presidential farm', 61],
        ['presidential farm', 75]],
    [['farm', 88], ['"la eminencia" farm', 947]],
    [['facilities', 1026]],
    [['tank truck', 1341], ['truck', 1346]]],
    'Victim': [[['cerezo', 139]]],
    'Weapon': []}}

    capitalize: whether to capitalize doctext or not
    '''

    res = {
        'docid': doc['docid'], 
        'document': doc['doctext'], # the raw text document.
        'annotation': [] # A list of templates. In role-filler entity extraction, we only have one template for each don't care about this.       
    }

    if capitalize:
        # split doctext into sentences
        sentences = sent_tokenizer.tokenize(doc['doctext'])
        capitalized_doctext = ' '.join([sent.capitalize() for sent in sentences])
        res['document'] = capitalized_doctext

    # process "\n\n" and "\n" https://github.com/xinyadu/grit_doc_event_entity/blob/master/data/muc/scripts/preprocess.py
    # paragraphs = doc['doctext'].split("\n\n")
    # paragraphs_no_n = []
    # for para in paragraphs:
    #     para = " ".join(para.split("\n"))
    #     paragraphs_no_n.append(para)
    # doc_text_no_n = " ".join(paragraphs_no_n)

    # TODO: add "tags" in the document
    # res['document'] = doc_text_no_n

    annotation = doc['extracts']
    for role, entities in annotation.items():
        # make sure entities is not an empty list
        if entities:
            # make sure res['annotation'] has one dictionary
            if len(res['annotation']) == 0:
                res['annotation'].append({})
            res['annotation'][0][role] = process_entities(entities)

    return res

if __name__ == '__main__':
    
    p = argparse.ArgumentParser("Convert GRIT input data into ours format.")
    
    p.add_argument('--input_path', type=str, help="input file in GRIT format.") 
    p.add_argument('--output_path',type=str, help="path to store the output json file.")
    p.add_argument('--capitalize',action="store_true", help="whether to capitalize the first char of each sentence")
    args = p.parse_args()

    with open(args.input_path, 'r') as f:
        grit_inputs = [json.loads(l) for l in f.readlines()]

    all_processed_doc = dict()

    # iterate thru and process all grit documents 
    for grit_doc in grit_inputs:
        
        processed = convert(grit_doc, args.capitalize)
        doc_id = processed.pop('docid')
        all_processed_doc[doc_id] = processed
    
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(all_processed_doc))