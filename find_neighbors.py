import faiss
import numpy as np
import torch

import pickle

def get_neighbor(index,query, n,tz):
    '''
    D: distance
    I: neighbor index, here is token index
    '''
    D, I = index.search(query, n)
    words = []
    for i in I[0]:
        # token index -> word
        words.append(tz.IdToToken(i))

    return words

def truncate_sequence(tokens):
    """
    Truncate sequence pair
    """
    max_num_tokens = 128-2-3
    while True:
        if len(tokens) <= max_num_tokens:
            break
        idx = 0 if random.random() < 0.5 else len(tokens) - 1
        tokens.pop(idx)


class Vocab_finder:
    def __init__(self):
        with open('/iesl/canvas/rueiyaosun/raw_val.pkl', 'rb') as handle:
            raw_text = pickle.load(handle)
        self.raw_text=raw_text
        self.total_examples =len(self.raw_text)

    def build_faiss(self, word_embed):
        self.index = faiss.index_factory(768, "Flat", faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(word_embed)
        self.index.add(word_embed)

    def get_facet(self,model,tokenizer,choose_id):
        sent = self.raw_text[choose_id]
        token = tokenizer.EncodeAsIds(sent).tokenization
        truncate_sequence(token)
        token = [tokenizer.get_command('ENC').Id] + [tokenizer.get_command('s_1').Id] + [
            tokenizer.get_command('s_2').Id] + [tokenizer.get_command('s_3').Id] + token
        sent_encode = model.model.bert(torch.tensor([token]).cuda(), output_all_encoded_layers=False)[0]
        view_list=[]
        for i in range(1, 4):
            tmp = model.model.sent['mf']['v_' + str(i)](sent_encode[:, i])
            view = model.model.sent['mf']['s_' + str(i)](tmp,not_norm=model.model.unnorm_facet)
            view = view.cpu().detach().numpy()
            faiss.normalize_L2(view)
            view_list.append(view)

        return view_list

    def query(self,view_list,choose_id,tz):
        n=5
        print('Query:', self.raw_text[choose_id])
        print(get_neighbor(self.index,view_list[0], n,tz))
        print(get_neighbor(self.index,view_list[1], n,tz))
        print(get_neighbor(self.index,view_list[2], n,tz))

