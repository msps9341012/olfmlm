import os
import random
import numpy as np
import psutil
import torch
import pickle
import sys

from olfmlm.configure_data import configure_data
from olfmlm.model import BertModel
from olfmlm.optim import Adam
from olfmlm.utils import save_checkpoint
from olfmlm.utils import load_checkpoint
from argparse import Namespace

from olfmlm.paths import pretrained_path

from tqdm import tqdm

mode=sys.argv[1]
model_path = pretrained_path +"_{mode}/mf+mlm/best/model.pt".format(mode = mode)

# The path where you want to store the gradient of facet vectors.
save_dir='/iesl/canvas/rueiyaosun/grads'



args=Namespace(alternating=False, always_mlm=True, attention_dropout=0.1, batch_size=16, bert_config_file='bert_config.json', cache_dir='cache_dir', checkpoint_activations=False, clip_grad=1.0, continual_learning=False, cuda=True, delim=',', distributed_backend='nccl', dynamic_loss_scale=True, epochs=32, eval_batch_size=None, eval_iters=2000, eval_max_preds_per_seq=None, eval_seq_length=None, eval_text_key=None, eval_tokens=1000000, fp32_embedding=False, fp32_layernorm=False, fp32_tokentypes=False, hidden_dropout=0.0, hidden_size=1024, incremental=False, intermediate_size=None, layernorm_epsilon=1e-12, lazy_loader=True, load=None, load_all_rng=False, load_optim=True, load_rng=True, local_rank=None, log_interval=1000000, loose_json=False, lr=0.0001, lr_decay_iters=None, lr_decay_style='linear', max_dataset_size=None, max_position_embeddings=512, max_preds_per_seq=80, model_type='rg+mlm', modes='mlm,rg', no_aux=False, num_attention_heads=16, num_layers=24, num_workers=22, presplit_sentences=True, pretrained_bert=False, rank=0, resume_dataloader=False, save='pretrained_berts/rg+mlm', save_all_rng=False, save_iters=None, save_optim=True, save_rng=True, seed=1234, seq_length=128, shuffle=True, split='1000,1,1', test_data=None, text_key='text', tokenizer_model_type='bert-base-uncased', tokenizer_path='tokenizer.model', tokenizer_type='BertWordPieceTokenizer', track_results=True, train_data=['bert_corpus'], train_iters=1000000, train_tokens=1000000000, use_tfrecords=False, valid_data=None, vocab_size=30522, warmup=0.01, weight_decay=0.02, world_size=1)

args.pretrained_bert=True

args.modes='mlm,mf'
args.model_type='mf+mlm'

args.extra_token=mode.split('_')[1]
args.agg_function=mode.split('_')[0]

data_config = configure_data()
data_config.set_defaults(data_set_type='BERT', transpose=False)
(train_data, val_data, test_data), tokenizer = data_config.apply(args)
args.data_size = tokenizer.num_tokens


extracted_grads_word=[]
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads_word.append(grad_out[0])

# add hooks for embeddings, only add a hook to encoder wordpiece embeddings (not position)
def add_hooks(model):
    model.model.bert.embeddings.word_embeddings.weight.requires_grad = True
    model.model.bert.embeddings.word_embeddings.register_backward_hook(extract_grad_hook)

model = BertModel(tokenizer, args)
model_sd = torch.load(model_path, map_location='cpu')
model.load_state_dict(model_sd['sd'],strict=False)
add_hooks(model)
model.eval()

def truncate_sequence(tokens):
    """
    Truncate sequence pair
    """
    max_num_tokens = val_data.dataset.max_seq_len-2-3
    while True:
        if len(tokens) <= max_num_tokens:
            break
        idx = 0 if random.random() < 0.5 else len(tokens) - 1
        tokens.pop(idx)

from collections import defaultdict


v_dict=defaultdict(list)
word_embedding = model.model.bert.embeddings.word_embeddings.weight

with tqdm(total=159973) as pbar:
    for doc in val_data.dataset.ds:
        for sent in val_data.dataset.sentence_split(doc):
            token=tokenizer.EncodeAsIds(sent).tokenization
            truncate_sequence(token)
            token=[tokenizer.get_command('ENC').Id] + [tokenizer.get_command('s_1').Id] +[tokenizer.get_command('s_2').Id] + [tokenizer.get_command('s_3').Id] + token + [tokenizer.get_command('sep').Id]
            sent_encode=model.model.bert(torch.tensor([token]),output_all_encoded_layers=False)[0]
            for i in range(1,4):
                tmp=model.model.sent['mf']['v_'+str(i)](sent_encode[:,i])
                view=model.model.sent['mf']['s_'+str(i)](tmp)
                loss=torch.norm(view)
                if i<3:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward() #fire graph
                grads=extracted_grads_word[0][0][4:-1]
                token_emb = word_embedding[token[4:-1]]
                dot=(grads*token_emb).sum(dim=1)
                dot=dot/dot.sum()
                dot-=dot.min()
                dot/=dot.max()
                #grad_norm=torch.norm(grads,dim=1)/torch.norm(grads,dim=1).sum()
                v_dict[i].append(dot.cpu().detach().numpy())
                model.zero_grad()
                extracted_grads_word=[]
            pbar.update(1)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    

save_path=os.path.join(save_dir, mode)
if not os.path.exists(save_path):
    os.makedirs(save_path)


for i in range(1,4):
    save_file=os.path.join(save_path, 'view_'+str(i))
    with open(save_file, 'wb') as handle:
        pickle.dump(v_dict[i], handle, protocol=pickle.HIGHEST_PROTOCOL)








