[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/document-level-entity-based-extraction-as/role-filler-entity-extraction-on-muc-4)](https://paperswithcode.com/sota/role-filler-entity-extraction-on-muc-4?p=document-level-entity-based-extraction-as) </br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/document-level-entity-based-extraction-as/binary-relation-extraction-on-scirex)](https://paperswithcode.com/sota/binary-relation-extraction-on-scirex?p=document-level-entity-based-extraction-as) </br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/document-level-entity-based-extraction-as/4-ary-relation-extraction-on-scirex)](https://paperswithcode.com/sota/4-ary-relation-extraction-on-scirex?p=document-level-entity-based-extraction-as)</br>
# TempGen
Source code for the EMNLP' 21 paper [Document-level Entity-based Extraction as Template Generation](https://arxiv.org/abs/2109.04901).

## Dependencies 

All the required packages are listed in `requirements.txt`. To install all the dependencies, run

```
conda create -n tg python=3.7
conda activate tg
pip install -r requirements.txt
```


## Data

All data lies in directory `./data`. The processed REE output can be found at `data/muc34/proc_output/`. Files name with patterns `ree_*.json` refers to the train, dev, and test set data for role-filler entity extraction in our in-house representation. These files are converted from `grit_*.json`, which are the train, dev, and test copied from [GRIT's repo](https://github.com/xinyadu/grit_doc_event_entity/). The conversion script is `convert_grit.py`. An example of converting GRIT data into our in-house format is:

```
python convert_grit.py --input_path data/muc34/proc_output/grit_train.json --output_path data/muc34/proc_output/ree_train.json
```

As for SciREX, we downloaded the original dataset `data/scirex/release_data.tar.gz` from [the original SciREX repo](https://github.com/allenai/SciREX/tree/master/scirex_dataset). The extracted train, dev, and test files are located in `data/scirex/release_data`. These original data are transformed into our internal representations using `raw_scripts/process_scirex.sh` and stored in `data/scirex/proc_output`. The binary RE data does not have any post-fix, while the 4-ary RE data are post-fixxed with `_4ary`.

### Pre-processing

We adpated some of the pre-processing code from [Du et al. 2021](https://arxiv.org/abs/2008.09249). To produce our training data, you need to navigate to `raw_script` and extract documents by running

```
bash go_proc_doc.sh
```

with __Python 2.7 !!__ . (Previous works use Python 2.7 for this step of pre-processing. Will upgrade this script later when I have time.)

Then, use __Python 3.6__ or above to run the second pre-processing script for combining annotation and doucments. 

```
bash process_all_keys.sh
```

Please refer to the `raw_script/READMD.md` for more details about the data format.


## Training

Our formulation of document-level IE as template generation tasks allows the same model architecture applicable for role-filler entity extraction, binary relation extraction, and 4-ary relation extraction. Therefore, the same script `train.py` can be used for training models for all three tasks. The only difference in training models each task task is the config file.

Role-filler entity extraction
```
python train.py -c config/ree_generative_model.json
```
Binary relation extraction
```
python train.py -c config/bre_generative_model.json
```
4-ary relation extraction
```
python train.py -c config/4re_generative_model.json
```

The key difference between these two config files is the `task` field. Event template extraction has `task: ete`, while role-filler entity extraction has `task: ree`. To enable/ disable the Topk Copy mechanism, set `use_copy` to `true/ false`. 




## Evaluation

The evaluation scripts for MUC-4 REE and SciREX RE are `ree_eval.py` and `scirex_eval.py`, which are copied over from the [GRIT repo](https://github.com/xinyadu/grit_doc_event_entity/) and the [SciREX repo](https://github.com/allenai/SciREX).

To run evaluation on trained models, execute the `evaluate.py` script as follows:
```
python evaluate.py --gpu 0 --checkpoint $PATH_TO_MODEL/best.mdl
```
passing `--gpu -1` can run evaluation on CPUs.

The trained models can be downloaded from [here](https://drive.google.com/drive/folders/1D6-0mM7n3JeqXzspBtNWi6fQC4mJHdSb?usp=sharing) for reproduction purposes.

The structure of this repo is based on OneIE (https://blender.cs.illinois.edu/software/oneie/)

## Citation
```bibtex
@inproceedings{huang-etal-2021-tempgen,
    title = "Document-level Entity-based Extraction as Template Generation",
    author = "Huang, Kung-Hsiang  and
      Tang, Sam  and
      Peng, Nanyun",
    booktitle = "The 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```
