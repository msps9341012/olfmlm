import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from olfmlm.model.modeling import *
from collections import Counter
import pickle


# def set_requires_grad(module, val):
#     for p in module.parameters():
#         p.requires_grad = val

def get_freq_weight():
    with open('/iesl/canvas/rueiyaosun/freq_counter.pkl', 'rb') as handle:
        freq = pickle.load(handle) #only include vocab count>1, so the size is smaller than 30522
    freq.update(list(range(30522))) #bert vocab size
    freq = list(dict(sorted(freq.items(), key=lambda i: i[0])).values())
    freq = np.array(freq) - 1
    prob_w = freq / sum(freq)
    prob_w = 1e-4 / (prob_w + 1e-4)
    prob_w[102] = 0 #zero out [SEP]
    return prob_w

class BertSentHead(nn.Module):
    def __init__(self, config, num_classes=2):
        super(BertSentHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, num_classes)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertHeadTransform(nn.Module):
    def __init__(self, config, input_size=None):
        super(BertHeadTransform, self).__init__()
        input_size = input_size if input_size else config.hidden_size
        self.dense = nn.Linear(input_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMTokenHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, input_size=None):
        super(BertLMTokenHead, self).__init__()
        self.transform = BertHeadTransform(config, input_size=input_size)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertTokenHead(nn.Module):
    def __init__(self, config, num_classes=2, input_size=None):
        super(BertTokenHead, self).__init__()
        input_size = input_size if input_size else config.hidden_size
        self.transform = BertHeadTransform(config, input_size=input_size)
        self.decoder = nn.Linear(config.hidden_size, num_classes)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        predictions = self.decoder(hidden_states)
        return predictions

class BertPoolerforview(nn.Module):
    def __init__(self, config):
        super(BertPoolerforview, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, token_tensor):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        
        pooled_output = self.dense(token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    
    
class Bert(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, modes=["mlm"], extra_token='cls', agg_function='max'):
        super(Bert, self).__init__(config)
        self.bert = BertModel(config)
        self.lm = BertLMTokenHead(config, self.bert.embeddings.word_embeddings.weight)
        
        self.sent = torch.nn.ModuleDict()
        self.tok = torch.nn.ModuleDict()
        if "nsp" in modes:
            self.sent["nsp"] = BertSentHead(config, num_classes=2)
        if "psp" in modes:
            self.sent["psp"] = BertSentHead(config, num_classes=3)
        if "sd" in modes:
            self.sent["sd"] = BertSentHead(config, num_classes=3)
        if "so" in modes:
            self.sent["so"] = BertSentHead(config, num_classes=2)
        if "sc" in modes:
            self.sent["sc"] = BertSentHead(config, num_classes=2)
        if "sbo" in modes:
            self.tok["sbo"] = BertLMTokenHead(config, self.bert.embeddings.word_embeddings.weight,
                                              input_size=config.hidden_size * 2)
        if "cap" in modes:
            self.tok["cap"] = BertTokenHead(config, num_classes=2)
        if "wlen" in modes:
            self.tok["wlen"] = BertTokenHead(config, num_classes=1)
        if "tf" in modes:
            self.tok["tf"] = BertTokenHead(config, num_classes=1)
        if "tf_idf" in modes:
            self.tok["tf_idf"] = BertTokenHead(config, num_classes=1)
        if "tc" in modes:
            self.tok["tc"] = BertTokenHead(config, num_classes=2)
        if "rg" in modes:
            self.sent["rg"] = BertHeadTransform(config)
        '''
        Define layers needed for our mf tasks
        '''
        if "mf" in modes:
            self.sent["mf"] = torch.nn.ModuleDict()
            self.sent["mf"]['s_1']=BertHeadTransform(config)
            self.sent["mf"]['s_2']=BertHeadTransform(config)
            self.sent["mf"]['s_3']=BertHeadTransform(config)
            self.sent['mf']['extra_head']=BertHeadTransform(config)
            self.extra_token = extra_token
            self.agg_function = agg_function
            self.sent["mf"]['v_1']=BertPoolerforview(config)
            self.sent["mf"]['v_2']=BertPoolerforview(config)
            self.sent["mf"]['v_3']=BertPoolerforview(config)
            self.sent["mf"]['extra_pool']=BertPoolerforview(config)
            self.sent["mf"]['weighted'] = nn.Linear(config.hidden_size*2, 1)
            self.softmax_weight=None
            self.prob_w = torch.tensor(get_freq_weight(),dtype=torch.float32).cuda()
            self.prob_w = torch.stack([self.prob_w]*32)

        if "fs" in modes:
            self.sent["fs"] = BertHeadTransform(config)
            self.tok["fs"] = BertHeadTransform(config)
        if "tgs" in modes:
            self.tok["tgs"] = BertTokenHead(config, num_classes=6, input_size=config.hidden_size * 3)
        self.apply(self.init_bert_weights)

    def forward(self, modes, input_ids, token_type_ids=None, task_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, checkpoint_activations=False):
        # assert len(input_ids) * len(token_type_ids) * len(attention_mask) == 1
        token_type_ids = token_type_ids if token_type_ids is None else torch.cat(token_type_ids, dim=0)
        task_ids = task_ids if task_ids is None else torch.cat(task_ids, dim=0)
        att_mask = attention_mask if attention_mask is None else torch.cat(attention_mask, dim=0)
        sequence_output, pooled_output = self.bert(torch.cat(input_ids, dim=0), token_type_ids, task_ids, att_mask,
                                                   output_all_encoded_layers=False,
                                                   checkpoint_activations=checkpoint_activations)
         
        scores = {}

        if "mlm" in modes:
            scores["mlm"] = self.lm(sequence_output)
        if "nsp" in modes:
            scores["nsp"] = self.sent["nsp"](pooled_output)
        if "psp" in modes:
            scores["psp"] = self.sent["psp"](pooled_output)
        if "sd" in modes:
            scores["sd"] = self.sent["sd"](pooled_output)
        if "so" in modes:
            scores["so"] = self.sent["so"](pooled_output)
            
        if "mf" in modes:
            
            half = len(input_ids[0])
            send_emb_list=[]
            recv_emb_list=[]
            #self.embedding_var_before_pool=self.var_of_embedding(sequence_output[:,1:4,:])
            pool_output_list=[]
            
            l=[] #for weighted sum, now is not our main target
            r=[] #for weighted sum, now is not our main target

            for i in range(1,4):
                l.append(sequence_output[:,i][:half])
                r.append(sequence_output[:,i][half:])
                '''
                Get facet by index and pass through pooler layer and transform layer
                '''
                pooled_facet=self.sent['mf']['v_'+str(i)](sequence_output[:,i])
                send_emb, recv_emb = pooled_facet[:half], pooled_facet[half:]
                send_emb, recv_emb = self.sent['mf']['s_'+str(i)](send_emb), self.sent['mf']['s_'+str(i)](recv_emb)
                send_emb_list.append(send_emb)
                recv_emb_list.append(recv_emb)
             
            #self.embedding_var_after_pool=self.var_of_embedding(torch.stack(pool_output_list).transpose(0,1))
            
            
            send_emb_tensor=torch.stack(send_emb_list)
            recv_emb_tensor=torch.stack(recv_emb_list)
            
            send_recv_list=[send_emb_tensor, recv_emb_tensor]
            
            #compute the variance between the facets
            self.emedding_var_after_trans=self.var_of_embedding(torch.cat(send_recv_list,dim=1).transpose(0,1),dim=1)
            self.emedding_var_across = self.var_of_embedding(torch.cat(send_recv_list, dim=1).transpose(0, 1),dim=0)

            '''
            Code related to dropout
            if np.random.uniform()<=0.5:
                 choice = np.random.randint(3, size=1).item()
                 send_emb_tensor = send_emb_tensor[choice].unsqueeze(dim=0)
                 recv_emb_tensor = recv_emb_tensor[choice].unsqueeze(dim=0)
            '''

            #get the frequency weight by token index
            token_index = torch.cat(input_ids, dim=0)
            freq_w = torch.gather(self.prob_w, 1, index=token_index)


            '''
            compute similarity score matrix corresponding to different choices
            '''

            if self.extra_token=='token':
                token_hidden = sequence_output[:, 4:, :]
                token_hidden_proj = self.lm.transform(token_hidden)

                score_left = self.get_probs_hidden(send_emb_tensor, token_hidden_proj[half:])
                score_right = self.get_probs_hidden(recv_emb_tensor, token_hidden_proj[:half])

                score_left = self.agg_function_map(torch.stack(score_left), self.agg_function, dim=0)
                score_right = self.agg_function_map(torch.stack(score_right), self.agg_function, dim=0)

                # score_left = self.word2vec_loss(score_left, att_mask[half:, :], freq_w[half:])
                # score_right  = self.word2vec_loss(score_right, att_mask[:half, :],freq_w[:half])

                freq_w = freq_w[:, 4:]
                score_left = self.masked_softmax(score_left,  att_mask[half:, :],
                                                 reduce_func='sum',word_weight=freq_w[half:])
                score_right = self.masked_softmax(score_right, att_mask[:half, :],
                                                  reduce_func='sum',word_weight=freq_w[:half])



            elif self.extra_token=='vocab':
                #bert vocab embedding by token index
                token_embeds = self.bert.embeddings.word_embeddings(token_index[:, 4:])

                score_left = self.get_probs_hidden(send_emb_tensor, token_embeds[half:])
                score_right = self.get_probs_hidden(recv_emb_tensor, token_embeds[:half])

                score_left = self.agg_function_map(torch.stack(score_left), self.agg_function, dim=0)
                score_right = self.agg_function_map(torch.stack(score_right), self.agg_function, dim=0)

                '''
                if using word2vec loss, then comment out the below masked_softmax lines
                
                score_left = self.word2vec_loss(score_left, att_mask[half:, :], freq_w[half:])
                score_right = self.word2vec_loss(score_right, att_mask[:half, :],freq_w[:half])
                '''

                freq_w= freq_w[:,4:]
                score_left = self.masked_softmax(score_left,  att_mask[half:, :],
                                                 reduce_func='weighted',word_weight=freq_w[half:])
                score_right = self.masked_softmax(score_right, att_mask[:half, :],
                                                  reduce_func='weighted',word_weight=freq_w[:half])



                # left_index = input_ids[0]
                # right_index = input_ids[1]
                #
                # dot_left = self.get_dot_vocab(send_emb_tensor)
                # dot_right = self.get_dot_vocab(recv_emb_tensor)
                #
                # dot_left = self.agg_function_map(dot_left, self.agg_function, dim=0)
                # dot_right = self.agg_function_map(dot_right, self.agg_function, dim=0)
                #
                # score_left = self.vocab_prob(dot_left, right_index, att_mask[half:, :])
                # score_right = self.vocab_prob(dot_right, left_index, att_mask[:half, :])





            elif self.extra_token=='cls':
                # view_all = []
                # view_all.append(torch.cat([torch.stack(l), torch.stack(3 * [pooled_output[half:]])], dim=2))
                # view_all.append(torch.cat([torch.stack(r), torch.stack(3 * [pooled_output[:half]])], dim=2))
                # self.softmax_weight = torch.cat(view_all).transpose(0, 1)
                pooled_out_trans = self.sent["mf"]['extra_head'](pooled_output)

                score_left = self.cross_cos_sim(send_emb_tensor, pooled_out_trans[half:])
                score_right = self.cross_cos_sim(recv_emb_tensor, pooled_out_trans[:half])

                score_left = self.agg_function_map(score_left, self.agg_function, dim=0)
                score_right = self.agg_function_map(score_right, self.agg_function, dim=0)


            elif self.extra_token=='avg':
                all_words = sequence_output[:, 4:, :]
                length_mask = att_mask[:, 4:].unsqueeze(2)
                average_tokens = (all_words * length_mask).sum(dim=1) / length_mask.sum(dim=1)
                average_pool = self.sent["mf"]['extra_pool'](average_tokens)
                pooled_out_trans = self.sent["mf"]['extra_head'](average_pool)

                # view_all = []
                # view_all.append(torch.cat([torch.stack(l), torch.stack(3 * [average_tokens[half:]])], dim=2))
                # view_all.append(torch.cat([torch.stack(r), torch.stack(3 * [average_tokens[:half]])], dim=2))
                # self.softmax_weight = torch.cat(view_all).transpose(0, 1)

                score_left = self.cross_cos_sim(send_emb_tensor, pooled_out_trans[half:])
                score_right = self.cross_cos_sim(recv_emb_tensor, pooled_out_trans[:half])

                score_left = self.agg_function_map(score_left, self.agg_function, dim=0)
                score_right = self.agg_function_map(score_right, self.agg_function, dim=0)



            elif self.extra_token=='all':
                '''
                The primitive settings, have not maintained for a while.
                Can skip it.
                '''

                # #score_all=[]
                # view_all=[]
                # for i in range(3):
                #     for j in range(3):
                #         view_all.append(torch.cat([l[i],r[j]],dim=1))
                #
                # self.softmax_weight=torch.stack(view_all).transpose(0,1)
                #
                for i in range(3):
                    score_all.append(self.cross_cos_sim(recv_emb_tensor,send_emb_tensor[i]))

                score_left = self.agg_function_map(torch.stack(score_all), self.agg_function, dim=(0,1))
                score_right = None

            else:
                pass

            scores["mf"] = [score_left, score_right]

            #scores["mf"]=torch.stack(view_all)
            #for i in range(3):
                #score_all.append(self.cross_cos_sim(recv_emb_tensor,send_emb_tensor[i]))
             
            
            #scores["mf"]=torch.stack(score_all)
            
#           #torch.amax(torch.stack(score_all),dim=(0,1))
            
                
        if "rg" in modes:
            half = len(input_ids[0])
            send_emb, recv_emb = pooled_output[:half], pooled_output[half:]
            send_emb, recv_emb = self.sent["rg"](send_emb), self.sent["rg"](recv_emb)
            scores["rg"] = self.cosine_similarity(send_emb, recv_emb)
        if "fs" in modes:
            half = len(input_ids[0])
            prev_emb, next_emb = pooled_output[:half], pooled_output[half:]
            prev_emb, next_emb = self.sent["fs"](prev_emb), self.sent["fs"](next_emb)
            prev_words, next_words = sequence_output[:half], sequence_output[half:]
            prev_words, next_words = self.tok["fs"](prev_words), self.tok["fs"](next_words)
            s1 = self.batch_cos_sim(next_words, prev_emb) #torch.torch.sigmoid(torch.bmm(next_words, prev_emb[:, :, None]))
            s2 = self.batch_cos_sim(prev_words, next_emb) #torch.sigmoid(torch.bmm(prev_words, next_emb[:, :, None]))
            sim = torch.cat((s1, s2), dim=1).squeeze().view(-1)
            #ref = torch.zeros_like(sim)
            scores["fs"] = sim #torch.stack((ref, sim), dim=1)
        if "sbo" in modes:
            output_concats = [torch.cat((sequence_output[:, 0], sequence_output[:, 0], sequence_output[:, 0]), dim=-1)]
            output_concats += [torch.cat((sequence_output[:, 0], sequence_output[:, 0], sequence_output[:, 1]), dim=-1)]
            for i in range(2, sequence_output.shape[1]):
                output_concats += [torch.cat((sequence_output[:, i - 2], sequence_output[:, i - 1],
                                              sequence_output[:, i]), dim=-1)]
            output_concats += [torch.cat((sequence_output[:, i + 2], sequence_output[:, i + 2]), dim=-1)]
            output_concats = torch.stack(output_concats, dim=1)
            scores["sbo"] = self.tok["sbo"](output_concats)
        if "cap" in modes:
            scores["cap"] = self.tok["cap"](sequence_output)
        if "wlen" in modes:
            scores["wlen"] = self.tok["wlen"](sequence_output)
        if "tf" in modes:
            scores["tf"] = self.tok["tf"](sequence_output)
        if "tf_idf" in modes:
            scores["tf_idf"] = self.tok["tf_idf"](sequence_output)
        if "sc" in modes:
            scores["sc"] = self.sent["sc"](pooled_output)
        if "tc" in modes:
            scores["tc"] = self.tok["tc"](sequence_output)
        if "tgs" in modes:
            output_concats = [torch.cat((sequence_output[:, 0], sequence_output[:, 0]), dim=-1)]
            # output_concats += [torch.cat((sequence_output[:, 0], sequence_output[:, 0], sequence_output[:, 1]), dim=-1)]
            for i in range(1, sequence_output.shape[1]):
                output_concats += [torch.cat((sequence_output[:, i - 1], sequence_output[:, i]), dim=-1)]
            output_concats = torch.stack(output_concats, dim=1)
            scores["tgs"] = self.tok["tgs"](output_concats)

        return scores

    def cosine_similarity(self, a, b):
        "taken from https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re"
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def batch_cos_sim(self, a, b):
        a_norm = a / a.norm(dim=2)[:, :, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.bmm(a_norm, b_norm[:, :, None])

    def inner_product(self, a, b):
        return torch.mm(a, b.transpose(0, 1))
    
    def cross_cos_sim(self,a,b,norm=True):
        if norm:
            a_norm = a / a.norm(dim=2)[:, :, None]
            b_norm = b / b.norm(dim=1)[:, None]
        else:
            a_norm = a
            b_norm = b
        return torch.matmul(a_norm,b_norm.transpose(0, 1)).transpose(1,2)
    
    def var_of_embedding(self,inputs,dim):
        #(batch, facet, embedding)
        inputs_norm = inputs/inputs.norm(dim=2,keepdim=True)
        pred_mean = inputs_norm.mean(dim = dim, keepdim = True)
        loss_set_div = - torch.mean( (inputs_norm - pred_mean).norm(dim = 2))
        return loss_set_div




    def agg_function_map(self,score,method,dim):
        if method=='max':
            return torch.amax(score,dim=dim)
        elif method=='logsum':
            return torch.logsumexp(score,dim=dim)

        elif method=='softmax':
            dim = score.shape[-1]
            return torch.nn.functional.softmax(score, dim=dim).mean(dim=0)

        elif method=='concat':
            # send, recv = model.model.send_recv_list
            # send = send.transpose(1,0).reshape(score.shape[-1],-1)
            # recv = recv.transpose(1,0).reshape(score.shape[-1],-1)
            '''
            send = send.transpose(1,0).reshape(half,-1)
            recv = recv.transpose(1,0).reshape(half,-1)
            score = model.model.cosine_similarity(send, recv)
            '''
            pass

        elif method=='w_softmax':
            '''
            score=torch.nn.functional.softmax(score,dim=3)
            softmax_weight=model.model.sent['mf']['weighted'](model.model.softmax_weight) #16,9,1

            softmax_weight=torch.nn.functional.softmax(softmax_weight,dim=1).reshape(-1,3,3) #16,3,3
            softmax_weight=softmax_weight.transpose(0,2).unsqueeze(dim=3) #3,3,16,1

            score=torch.mul(score,softmax_weight).sum(dim=(0,1))

            score=torch.log(score)
            '''
            pass

        elif method=='hybrid':
            pass
        else:
            pass

    def word2vec_loss(self, dot, mask, word_weight):

        half=dot.shape[0]
        length=dot.shape[-1]
        freq_w_mask = word_weight*mask
        #freq_w_mask = mask
        freq_w_mask = freq_w_mask[:, 4:]

        neg_freq_w_sum = freq_w_mask.sum().expand(mask.shape[0]) - freq_w_mask.sum(dim=1)

        freq_w_mask = freq_w_mask.reshape(-1,1)
        dot = dot.reshape(half,-1,1)
        all=torch.ones(half*length,1).cuda()

        loss_list=[]
        for i in range(half):
            labels = torch.cat([torch.zeros(i*length),torch.ones(length),torch.zeros((half-i-1)*length)])
            labels = torch.autograd.Variable(labels).cuda()
            labels = labels.unsqueeze(dim=1)

            per_loss = nn.functional.binary_cross_entropy_with_logits(dot[i],labels,
                                                                      weight=freq_w_mask,reduction='none')
            pos_loss = (per_loss*labels).sum()

            neg_loss = (per_loss*(all-labels)).sum()
            neg_loss = neg_loss/neg_freq_w_sum[0]
            loss_list.append(pos_loss+neg_loss)

        return torch.stack(loss_list).mean()

    def masked_softmax(self, prob, mask, reduce_func=None, word_weight=None):
        '''
        mask out padding
        '''

        mask = mask[:, 4:]
        half = prob.shape[0]
        prob = prob * mask.float()
        if reduce_func == 'log':
            prob = torch.nn.functional.log_softmax(prob.reshape(half, -1), dim=1)
            prob = prob.reshape(half, half, -1)
            prob = (prob * mask).sum(dim=2) / mask.sum(dim=1)

        elif reduce_func=='weighted':
            prob = torch.nn.functional.log_softmax(prob.reshape(half, -1), dim=1)
            prob = prob.reshape(half, half, -1)
            prob = prob * word_weight
            prob = (prob * mask).sum(dim=2) / mask.sum(dim=1)

        else:
            prob = torch.nn.functional.softmax(prob.reshape(half, -1), dim=1)

            prob = prob.reshape(half, half, -1)

            prob = prob * mask.float()
            # normalize the probability again
            prob = prob / (prob.sum(dim=(1, 2), keepdim=True) + 1e-13)
            prob = prob.sum(dim=2)
            prob = torch.log(prob+1e-13)

        return prob



    def get_probs_hidden(self, facet_set, hidden_output):
        half = hidden_output.shape[0]
        #token_hidden = hidden_output
        token_hidden = hidden_output.detach().clone()
        score=[]
        
        for i in range(len(facet_set)):
            
            facet_i = facet_set[i]
            prob = torch.matmul(token_hidden, facet_i.T)
            prob = prob.transpose(1,2).transpose(1,0)

            score.append(prob)
        return score


    def get_dot_vocab(self, facet_set):
        token_embedding_weights = self.bert.embeddings.word_embeddings.weight#.detach().clone()
        facet_dot = torch.matmul(facet_set, token_embedding_weights.T)
        return facet_dot

    def vocab_prob(self,dot_prod, index_list, mask):
        half = index_list.shape[0]
        prob = torch.nn.functional.softmax(dot_prod)
        '''
        prob =torch.exp(dot_prod)
        speical_tokens = prob[:, 0:4].sum(dim=1, keepdim=True) + prob[:, 101:104].sum(dim=1, keepdim=True)
        prob = prob/(prob.sum(dim=1, keepdim=True) - speical_tokens) #normalize to prob
        '''
        prob = torch.log(prob)
        #prob = prob*self.prob_w
        
        prob = prob.expand(half, half, -1).transpose(1, 0)  #16,16,30522
        index_all = index_list.expand(half, half, -1)  #16,16,128
        index_all = index_all[:, :, 4:-1] #16,16,123
        mask_all  = mask[:, 4:-1].expand(half, half, -1)
        
        sel_prob = torch.gather(prob, dim=2, index=index_all)
        score = (sel_prob * mask_all).sum(dim=2)/mask_all.sum(dim=2)
        #score = (sel_prob * mask_all).sum(dim=2)
        return score
