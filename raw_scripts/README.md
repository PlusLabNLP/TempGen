# Proc scripts for the raw MUC

The pre-processing is composed of two parts: 

1. Process dcuments: This step loops over the corpus and generate a list of json file. Each json corresponds to a document. The script is adpated from https://github.com/xinyadu/grit_doc_event_entity/tree/master/data/muc/raw_files/raw_scripts. 

```
bash go_proc_doc.sh
```


2. Process anntation: This step gather all the annotation and the documents processed in the first step to create a json file for each split: `train.json`, `dev.json` and `test.json`. 

```
bash process_all_keys.sh
```

The json file generated has the following format:

```
{
    "doc_id":{
        "document": "...",
        "annotation": [
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

    },
    ...
}

```

__Note that you need Python 2.7 for the first step and Python 3.6 for the second!!__ Will fix this issue when I have time.

