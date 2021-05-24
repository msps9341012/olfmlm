Code for "On Losses for Modern Language Models" (ACL: https://www.aclweb.org/anthology/2020.emnlp-main.403/, arxiv: https://arxiv.org/abs/2010.01694)

This repository is primarily for reproducibility and posterity. It is not maintained.

Thank you to NVIDIA and NYU's jiant group for their code which helped create the base of this repo. Specifically
https://github.com/NVIDIA/Megatron-LM/commits/master (commit 0399d32c75b4719c89b91c18a173d05936112036)  
and  
https://github.com/nyu-mll/jiant/commits/master (commit 14d9e3d294b6cb4a29b70325b2b993d5926fe668)  
were used.

# Setup
Only tested on python3.6.

```
python -m pip install virtualenv
virtualenv bert_env
source bert_env/bin/activate
pip install -r requirements.txt
```


# TL;DR 
- Documentation
  - [Link](https://hackmd.io/@5ZnjJQRbT9CfVYeuVPNXVw/HylPupjNO)
  - Detail the whole project and the progresses we have made. 
    - Also point out the codes that I have modified. 
   
- Dataset
  - The preprocessed dataset is in ```/iesl/canvas/rueiyaosun/train_data/bert_corpus.lazy```. 
    - You can copy the dataset to a```train_data``` folder created in your location. 
    - And then set all the saving and loading paths in the ```paths.py```.
  - Also copy the ```/iesl/canvas/rueiyaosun/olfmlm/data_utils/random_index.npy``` to the same location in your workspace. 
    - This file is to make sure we will have the same training/dev/testing set.
  - Copy the word_frequency count file, ```/iesl/canvas/rueiyaosun/freq_counter.pkl``` to your space
    - Set your path in [here](https://github.com/msps9341012/olfmlm/blob/mf/model/new_models.py#L14)
    
- Commands for training.
  ```
  bash olfmlm/scripts/pretrain_bert.sh --model-type mf --pretrained-bert --save-iters 86400 --lr 2e-5 
              --agg-function max --warmup 0 --extra-token token --same-weight
  ```
    - This means using pretrain-bert weights, do not use warmup and training with mf task and initialize 3 facets' pooler and transform head weight the same.
      - Save the model every 86400 iterations (basically a day)
      - Change the save folder name in [here](https://github.com/msps9341012/olfmlm/blob/mf/arguments.py#L307) in case you want to try different setting
    - The choices for agg-function are 'max', 'logsum', 'softmax' (not stable). 
    - While for the extra-token, we have 'token', 'vocab', 'cls', 'avg' and 'all'.
    - Now in the logging iteration during **training**, the code will also do the nearest token finding
      - Update your path to ```raw_val.pkl``` in [here](https://github.com/msps9341012/olfmlm/blob/mf/find_neighbors.py#L34)
    - The remaining arguments are the same to the original repo. 
      - You can see more details descriptions in ```arguments.py```
    - Dropout code is in [here](https://github.com/msps9341012/olfmlm/blob/mf/model/new_models.py#L265-L271)
      - Make the model only consider one facet
      - _Have not tested it on our new setting, so might have some issues_
  
  

- Visualization
  - Get the output hidden states for all facets, you can change the ```save_dir``` within the file.
    - ```python -m olfmlm.extract_embedd max_token```
    - It will output three ```.npy file``` under the ```save_dir```.
  - Get the gradient for all facets _(can ignore now)_
    - ```python -m olfmlm.grad_analysis max_token```
  - Copy the raw text file of validation set from ```/iesl/canvas/rueiyaosun/raw_val.pkl``` to your workspace.  
  - Open the corresponding ```.ipynb``` file to do the visualization. These files are self-explanatory.
    - check_vocab.ipynb : find the nearest token
    - Analyze_emb.ipynb : find the nearest sentence
    - Analyze_grad.ipynb : visualize the important tokens in the sentence 
  
  Make sure to **move** those ```.ipynb``` files outside the ```olfmlm``` fold to prevent path from conflicts.

- Evaluation on Glue task
  - Get glue data in ```/iesl/canvas/rueiyaosun/glue_data```
    - change the ```data path``` in ```paths.py``` to make them linked
  - Run convert_state_dict.py to your saved checkedpoint to get converted state
    - ```python convert_state_dict.py pretrained_berts_max_token/mf+mlm/best/model.pt```
  - Use sbatch to excute submit.sh to finetune Bert model in three runs, the required arg is the suffix of your methd (the words after "pretrained_berts_")
    - ```sbatch submit.sh max_token```
    - Remember to change the workspace path in the sh file.
  - After three runs are done, run ```extract_result.sh max_token``` to show the score for each task.
    - Also need to change the workspace path in the sh file.


You can also create an account of ```comet_ml``` and change your api-key in the main function of ```pretrain_bert.py```. It is a free and powerful visualization tool.


# Usage
The code enables pre-training a transformer (size specified in bert_config.json) using any combination of the following tasks (aka modes/losses):
"mlm", "nsp", "psp", "sd", "so", "rg", "fs", "tc", "sc", "sbo", "wlen", "cap", "tf", "tf_idf", or "tgs". See paper for details regarding the modes.
NOTE: PSP (previous sentence prediction) is equivalent to ASP (adjacent sentence prediction) from the paper. RG (referential game) is equivalent to QT (quick thoughts variant) from the paper.

They can be combined using any of the following methods:
- Summing all losses (default, incompatible between a small subset of tasks, see paper for more detail)
- Continuous Multi-Task Learning, based on ERNIE 2.0 (--continual-learning True)
- Alternating between losses (--alternating True)

With the following modifiers:
- Always using MLM loss (--always-mlm True, which is the default and highly recommended, see paper for more details)
- Incrementally add tasks each epoch (--incremental)
- Use data formatting for tasks, but zero out losses from auxiliary tasks (--no-aux True, not recommended, used for testing)

Set paths to read/save/load from in paths.py

To create datasets, see data_utils/make_dataset.py

For tf_idf prediction, you need to first calculate the idf score for your dataset. See idf.py for a script to do this.

## Pre-training
To run pretraining :
`bash olfmlm/scripts/pretrain_bert.sh --model-type [model type]`
Where model type is the name of the model you want to train. If model type is one of the modes, it will train using mlm and that mode (if model type is mlm, it will train using just mlm).
The --modes argument will override this default behaviour. If model type is not a specified mode, the--modes argument is required.

## Distributed Pretraining
Use pretrain_bert_distributed.sh instead.
`bash olfmlm/scripts/pretrain_bert_distributed.sh --model-type [model type]`

## Evaluation
To run evaluation:
You will need to convert the saved state dict of the required model using the convert_state_dict.py file.
Then run:
`python3 -m olfmlm.evaluate.main --exp_name [experiment name]`
Where experiment name is the same as the model type above. If using a saved checkpoint instead of the best model, use the --checkpoint argument.



