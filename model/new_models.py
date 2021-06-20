import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from olfmlm.model.modeling import *
from collections import Counter
import pickle
from olfmlm.NNSC import MatrixReconstruction as MR


# def set_requires_grad(module, val):
#     for p in module.parameters():
#         p.requires_grad = val

def estimate_coeff_mat_batch_opt(target_embeddings, basis_pred, L1_losss_B, device, coeff_opt='rmsprop', lr=0.05, max_iter=60):
    batch_size = target_embeddings.size(0)
    mr = MR(batch_size, target_embeddings.size(1), basis_pred.size(1), device=device)
    loss_func = torch.nn.MSELoss(reduction='sum')

    # opt = torch.optim.LBFGS(mr.parameters(), lr=lr, max_iter=max_iter, max_eval=None, tolerance_grad=1e-05,
    #                         tolerance_change=1e-09, history_size=100, line_search_fn=None)
    #
    # def closure():
    #     opt.zero_grad()
    #     mr.compute_coeff_pos()
    #     pred = mr(basis_pred)
    #     loss = loss_func(pred, target_embeddings) / 2
    #     # loss += L1_losss_B * mr.coeff.abs().sum()
    #     loss += L1_losss_B * (mr.coeff.abs().sum() + mr.coeff.diagonal(dim1=1, dim2=2).abs().sum())
    #     # print('loss:', loss.item())
    #     loss.backward()
    #
    #     return loss
    #
    # opt.step(closure)

    if coeff_opt == 'sgd':
        opt = torch.optim.SGD(mr.parameters(), lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    elif coeff_opt == 'asgd':
        opt = torch.optim.ASGD(mr.parameters(), lr=lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    elif coeff_opt == 'adagrad':
        opt = torch.optim.Adagrad(mr.parameters(), lr=lr, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
    elif coeff_opt == 'rmsprop':
        opt = torch.optim.RMSprop(mr.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                                  centered=False)
    elif coeff_opt == 'adam':
        opt = torch.optim.Adam(mr.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        raise RuntimeError('%s not implemented for coefficient estimation. Please check args.' % coeff_opt)

    for i in range(max_iter):
        opt.zero_grad()
        pred = mr(basis_pred)
        loss = loss_func(pred, target_embeddings) / 2
        # loss += L1_losss_B * mr.coeff.abs().sum()
        # loss += L1_losss_B * (mr.coeff.abs().sum() + mr.coeff.diagonal(dim1=1, dim2=2).abs().sum())
        loss += L1_losss_B * mr.coeff.abs().sum()
        # print('loss:', loss.item())
        loss.backward()
        opt.step()
        mr.compute_coeff_pos()

    return mr.coeff.detach()


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
    def __init__(self, config, input_size=None, use_perturbation=False):
        super(BertHeadTransform, self).__init__()
        input_size = input_size if input_size else config.hidden_size
        self.dense = nn.Linear(input_size, config.hidden_size)
        if use_perturbation:
            self.perturbation = nn.Linear(input_size, config.hidden_size,bias=False)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states, not_norm=False):
        #merge the self.perturbation here
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        if not not_norm:
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

    def forward(self, hidden_states, not_norm=False):
        hidden_states = self.transform(hidden_states, not_norm=not_norm)
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

    def __init__(self, config, modes=["mlm"], extra_token='cls', agg_function='max',
                 unnorm_facet=False,unnorm_token=False, facet2facet=False,
                 use_dropout=False, autoenc_reg_const=0.0, num_facets=3):
        super(Bert, self).__init__(config)
        self.bert = BertModel(config)
        self.lm = BertLMTokenHead(config, self.bert.embeddings.word_embeddings.weight)

        self.unnorm_facet = unnorm_facet
        self.unnorm_token = unnorm_token
        self.facet2facet=facet2facet
        self.use_dropout = use_dropout
        self.num_facets = num_facets
        
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
            for i in range(1,self.num_facets+1):
                self.sent["mf"]['s_{}'.format(i)] = BertHeadTransform(config, use_perturbation=False)
                self.sent["mf"]['v_{}'.format(i)] = BertPoolerforview(config)
            #self.sent['mf']['extra_head']=BertHeadTransform(config)
            self.extra_token = extra_token
            self.agg_function = agg_function

            #self.sent["mf"]['extra_pool']=BertPoolerforview(config)
            #self.sent["mf"]['weighted'] = nn.Linear(config.hidden_size*2, 1)
            self.softmax_weight=None
            self.prob_w = torch.tensor(get_freq_weight(),dtype=torch.float32).cuda()
            #self.prob_w = torch.stack([self.prob_w]*32)
            self.autoenc_reg_const = autoenc_reg_const
            self.criterion_autoenc = torch.nn.MSELoss(reduction='mean')

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
            scores["mlm"] = self.lm(sequence_output, not_norm=self.unnorm_token)
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
            
            l = [] #for weighted sum, now is not our main target
            r = [] #for weighted sum, now is not our main target

            for i in range(1, self.num_facets+1):
                l.append(sequence_output[:,i][:half])
                r.append(sequence_output[:,i][half:])
                '''
                Get facet by index and pass through pooler layer and transform layer
                '''
                pooled_facet = self.sent['mf']['v_'+str(i)](sequence_output[:,i])
                send_emb, recv_emb = pooled_facet[:half], pooled_facet[half:]
                send_emb, recv_emb = self.sent['mf']['s_'+str(i)](send_emb, not_norm=self.unnorm_facet), self.sent['mf']['s_'+str(i)](recv_emb, not_norm=self.unnorm_facet)
                send_emb_list.append(send_emb)
                recv_emb_list.append(recv_emb)
             
            #self.embedding_var_after_pool=self.var_of_embedding(torch.stack(pool_output_list).transpose(0,1))


            send_emb_tensor=torch.stack(send_emb_list)
            recv_emb_tensor=torch.stack(recv_emb_list)


            self.corr_list = self.est_correlation(send_emb_tensor,recv_emb_tensor)

            send_recv_list=[send_emb_tensor, recv_emb_tensor]
            #compute the variance between the facets
            self.emedding_var_after_trans=self.var_of_embedding(torch.cat(send_recv_list,dim=1).transpose(0,1),dim=1)
            self.emedding_var_across = self.var_of_embedding(torch.cat(send_recv_list, dim=1).transpose(0, 1),dim=0)

            '''
            Init stat
            '''
            self.facet_agg_stats = {}
            if self.num_facets>0:
                if 'token' in self.extra_token:
                    self.facet_agg_stats['token'] = torch.zeros(self.num_facets)
                if 'vocab' in self.extra_token:
                    self.facet_agg_stats['vocab'] = torch.zeros(self.num_facets)
                if self.facet2facet:
                    self.facet_agg_stats['f2f'] = torch.zeros(self.num_facets**2)

            '''
            Code related to dropout
            '''
            drop_out_flag = False
            if self.use_dropout:
                if np.random.uniform()<=0.05:
                    choice = np.random.randint(self.num_facets, size=1).item()
                    #choice = 0
                    send_emb_tensor = send_emb_tensor[choice].unsqueeze(dim=0)
                    recv_emb_tensor = recv_emb_tensor[choice].unsqueeze(dim=0)
                    drop_out_flag = True
            #get the frequency weight by token index


            token_index = torch.cat(input_ids, dim=0)
            prob_w_tensor = torch.stack([self.prob_w] * (half * 2))
            freq_w = torch.gather(prob_w_tensor, 1, index=token_index)

            freq_w = freq_w[:, (self.num_facets + 1):]
            '''
            compute similarity score matrix corresponding to different choices
            '''

            if self.extra_token=='token':
                token_hidden = sequence_output[:, (self.num_facets+1):, :]
                token_hidden_proj = self.lm.transform(token_hidden,not_norm=self.unnorm_token)

                score_left = self.get_probs_hidden(send_emb_tensor, token_hidden_proj[half:])
                score_right = self.get_probs_hidden(recv_emb_tensor, token_hidden_proj[:half])

                score_left = self.agg_function_map(torch.stack(score_left), self.agg_function, dim=0)
                score_right = self.agg_function_map(torch.stack(score_right), self.agg_function, dim=0)

                # score_left = self.word2vec_loss(score_left, att_mask[half:, :], freq_w[half:])
                # score_right  = self.word2vec_loss(score_right, att_mask[:half, :],freq_w[:half])


                score_left = self.masked_softmax(score_left,  att_mask[half:, :],
                                                 reduce_func='weighted', word_weight=freq_w[half:])
                score_right = self.masked_softmax(score_right, att_mask[:half, :],
                                                  reduce_func='weighted', word_weight=freq_w[:half])

            elif self.extra_token=='token-mr':
                #(batch, number of tokens, emb_size)
                token_hidden = sequence_output[:, (self.num_facets+1):, :]
                #project to embedding space
                token_hidden_proj = self.lm.transform(token_hidden, not_norm=self.unnorm_token)

                #normalize
                token_hidden_proj_detach = token_hidden_proj.detach().clone()
                #token_hidden_proj_detach = token_hidden_proj_detach / (0.000000000001 + token_hidden_proj_detach.norm(dim=2,keepdim=True))  # If this step is really slow, consider to do normalization before doing unfold

                #compute mr matrix for pos/neg ample, return dot-product
                score_left = self.mr_loss(token_hidden_proj_detach[half:], send_emb_tensor)
                score_right = self.mr_loss(token_hidden_proj_detach[:half], recv_emb_tensor)

                score_left = self.masked_softmax(score_left,  att_mask[half:, :],
                                                 reduce_func='weighted', word_weight=freq_w[half:])
                score_right = self.masked_softmax(score_right, att_mask[:half, :],
                                                  reduce_func='weighted', word_weight=freq_w[:half])
                #mr_neg = estimate_coeff_mat_batch_opt(token_hidden_proj_detach[:half], recv_emb_tensor.detach().clone(), 0.2, torch.device('cuda'))
                #estimate_coeff_mat_batch_opt()

            elif self.extra_token=='vocab-mr':
                token_embeds = self.bert.embeddings.word_embeddings(token_index[:, (self.num_facets+1):])
                token_embeds_detach = token_embeds.detach().clone()
                #token_embeds_detach = token_embeds_detach / (0.000000000001 + token_embeds_detach.norm(dim=2, keepdim=True))  # If this step is really slow, consider to do normalization before doing unfold

                score_left = self.mr_loss(token_embeds_detach[half:], send_emb_tensor)
                score_right = self.mr_loss(token_embeds_detach[:half], recv_emb_tensor)

                score_left = self.masked_softmax(score_left,  att_mask[half:, :],
                                                 reduce_func='weighted', word_weight=freq_w[half:])
                score_right = self.masked_softmax(score_right, att_mask[:half, :],
                                                  reduce_func='weighted', word_weight=freq_w[:half])

            elif self.extra_token=='vocab':
                #bert vocab embedding by token index
                token_embeds = self.bert.embeddings.word_embeddings(token_index[:, (self.num_facets+1):])

                score_left = self.get_probs_hidden(send_emb_tensor, token_embeds[half:])
                score_right = self.get_probs_hidden(recv_emb_tensor, token_embeds[:half])

                score_left = self.agg_function_map(torch.stack(score_left), self.agg_function, dim=0)
                score_right = self.agg_function_map(torch.stack(score_right), self.agg_function, dim=0)

                '''
                if using word2vec loss, then comment out the below masked_softmax lines
                
                score_left = self.word2vec_loss(score_left, att_mask[half:, :], freq_w[half:])
                score_right = self.word2vec_loss(score_right, att_mask[:half, :],freq_w[:half])
                '''

                score_left = self.masked_softmax(score_left,  att_mask[half:, :],
                                                 reduce_func='weighted', word_weight=freq_w[half:])
                score_right = self.masked_softmax(score_right, att_mask[:half, :],
                                                  reduce_func='weighted', word_weight=freq_w[:half])



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
                all_words = sequence_output[:, (self.num_facets+1):, :]
                length_mask = att_mask[:, (self.num_facets+1):].unsqueeze(2)
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

            else:
                score_left=0
                score_right=0



            if self.facet2facet:

                # #score_all=[]
                # view_all=[]
                # for i in range(3):
                #     for j in range(3):
                #         view_all.append(torch.cat([l[i],r[j]],dim=1))
                #
                # self.softmax_weight=torch.stack(view_all).transpose(0,1)
                #
                if drop_out_flag:
                    score_all = self.cosine_similarity(send_emb_tensor[0], recv_emb_tensor[0])

                else:
                    score_all = []
                    for i in range(self.num_facets):
                        score_all.append(self.cross_cos_sim(recv_emb_tensor,send_emb_tensor[i]))

                    score_all = self.agg_function_map(torch.stack(score_all), self.agg_function, dim=(0,1))

            else:
                score_all = None

            if self.autoenc_reg_const > 0:
                # TODO: temporary to turn off dropout on autoencoder.
                drop_out_flag = False
                if drop_out_flag:
                    facets_avg = sequence_output[:,choice,:]
                else:
                    facets_avg = torch.mean(sequence_output[:,1:(self.num_facets+1),:], dim=1)

                token_embeds = self.bert.embeddings.word_embeddings(token_index[:,4:]).detach().clone()
                # I have made a change in my version so that it is like this, but I believe on your end
                # you do not need to slice freq_w here.
                token_weights = (freq_w * att_mask[:,(self.num_facets+1):]).unsqueeze(-1)
                words_weighted_sum = (token_embeds * token_weights).mean(dim=1)

                loss_autoenc = self.autoenc_reg_const * self.criterion_autoenc(facets_avg, words_weighted_sum)

            else:
                loss_autoenc = torch.tensor(0.).cuda()

            scores["mf"] = [score_left, score_right, score_all, loss_autoenc]
            '''
            Normalize the stats
            '''

            if 'token' in self.extra_token:
                self.facet_agg_stats['token'] = self.facet_agg_stats['token'] / torch.sum(self.facet_agg_stats['token'])
            if 'vocab' in self.extra_token:
                self.facet_agg_stats['vocab'] = self.facet_agg_stats['vocab'] / torch.sum(self.facet_agg_stats['vocab'])
            if self.facet2facet:
                self.facet_agg_stats['f2f'] = self.facet_agg_stats['f2f'] / torch.sum(self.facet_agg_stats['f2f'])


                
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
    
    def cross_cos_sim(self, a, b, norm=True):
        if norm:
            a_norm = a / a.norm(dim=2)[:, :, None]
            b_norm = b / b.norm(dim=1)[:, None]
        else:
            a_norm = a
            b_norm = b
        return torch.matmul(a_norm,b_norm.transpose(0, 1)).transpose(1,2)
    
    def var_of_embedding(self, inputs, dim):
        #(batch, facet, embedding)
        inputs_norm = inputs/inputs.norm(dim=2, keepdim=True)
        pred_mean = inputs_norm.mean(dim=dim, keepdim=True)
        loss_set_div = - torch.mean( (inputs_norm - pred_mean).norm(dim=2))
        return loss_set_div




    def agg_function_map(self, score, method, dim):
        num_facets = len(score)
        if method == 'max':
            if isinstance(dim, int):  #case token or vocab
                vals, idxs = torch.max(score, dim=dim)
                if num_facets == self.num_facets:
                    counts = torch.bincount(idxs.flatten().cpu(), minlength=self.num_facets)
                    if self.extra_token == 'token':
                        self.facet_agg_stats['token'] += counts
                    elif self.extra_token == 'vocab':
                        self.facet_agg_stats['vocab'] += counts
                # else dropout batch, and stats will appear as 'nan'
                return vals
            elif isinstance(dim, tuple):  #case facet2facet. Assumes dim=(0,1)
                vals_intermed, idxs_intermed = torch.max(score, dim=0)  # max over dim 0 of score
                vals_final, idxs_final = torch.max(vals_intermed, dim=0)  # max over dims 0 and 1 of score
                if num_facets == self.num_facets:
                    second_facet_idxs = idxs_final.unsqueeze(0)
                    first_facet_idxs = torch.gather(idxs_intermed, dim=0, index=second_facet_idxs)
                    flattened = (first_facet_idxs * self.num_facets + second_facet_idxs).flatten().cpu()
                    counts = torch.bincount(flattened, minlength=self.num_facets**2)
                    self.facet_agg_stats['f2f'] += counts
                # else dropout batch, and stats will appear as 'nan'
                return vals_final
            else:
                pass
                #return torch.amax(score,dim=dim)
        elif method == 'logsum':
            return torch.logsumexp(score,dim=dim)

        elif method == 'softmax':
            dim = score.shape[-1]
            return torch.nn.functional.softmax(score, dim=dim).mean(dim=0)

        elif method == 'concat':
            # send, recv = model.model.send_recv_list
            # send = send.transpose(1,0).reshape(score.shape[-1],-1)
            # recv = recv.transpose(1,0).reshape(score.shape[-1],-1)
            '''
            send = send.transpose(1,0).reshape(half,-1)
            recv = recv.transpose(1,0).reshape(half,-1)
            score = model.model.cosine_similarity(send, recv)
            '''
            pass

        elif method == 'w_softmax':
            '''
            score=torch.nn.functional.softmax(score,dim=3)
            softmax_weight=model.model.sent['mf']['weighted'](model.model.softmax_weight) #16,9,1

            softmax_weight=torch.nn.functional.softmax(softmax_weight,dim=1).reshape(-1,3,3) #16,3,3
            softmax_weight=softmax_weight.transpose(0,2).unsqueeze(dim=3) #3,3,16,1

            score=torch.mul(score,softmax_weight).sum(dim=(0,1))

            score=torch.log(score)
            '''
            pass

        elif method == 'hybrid':
            pass
        else:
            pass

    def word2vec_loss(self, dot, mask, word_weight):

        half = dot.shape[0]
        length = dot.shape[-1]
        freq_w_mask = word_weight*mask
        #freq_w_mask = mask
        freq_w_mask = freq_w_mask[:, self.num_facets+1:]

        neg_freq_w_sum = freq_w_mask.sum().expand(mask.shape[0]) - freq_w_mask.sum(dim=1)

        freq_w_mask = freq_w_mask.reshape(-1,1)
        dot = dot.reshape(half,-1,1)
        all = torch.ones(half*length,1).cuda()

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
        prob: (batch,batch,number_of_tokens)
        mask: (batch, number_of_tokens (include facet) )
        word_weight: (batch, number_of_tokens)
        '''



        mask = mask[:, self.num_facets+1:]
        mask = mask.float()
        word_weight = word_weight * mask

        #normalize
        #word_weight = word_weight / word_weight.sum(dim=1, keepdim=True)
        half = prob.shape[0] #batch size
        prob = prob * mask#.float()

        #prob = prob + (mask.expand(half, half, -1) + 1e-45).log()
        #prob = torch.where(mask.expand(half, half, -1) != 0, prob, rep_value.cuda())
        if reduce_func == 'log':
            prob = torch.nn.functional.log_softmax(prob.reshape(half, -1), dim=1)
            prob = prob.reshape(half, half, -1)
            prob = (prob * mask).sum(dim=2) / mask.sum(dim=1)

        elif reduce_func=='weighted':
            prob = torch.nn.functional.log_softmax(prob.reshape(half, -1), dim=1) #do softmax per sentences
            prob = prob.reshape(half, half, -1)
            prob = prob * word_weight #already masked
            prob = prob.sum(dim=2) / mask.sum(dim=1)

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
        index_all = index_all[:, :, self.num_facets+1:-1] #16,16,123
        mask_all  = mask[:, self.num_facets+1:-1].expand(half, half, -1)
        
        sel_prob = torch.gather(prob, dim=2, index=index_all)
        score = (sel_prob * mask_all).sum(dim=2)/mask_all.sum(dim=2)
        #score = (sel_prob * mask_all).sum(dim=2)
        return score

    def mr_loss(self,token_hidden_proj_detach,facet_emb):
        '''
        token_hidden_proj_detach: (batch, number_of_tokens, emb_size)
        facet_emb: (number_of_facet,batch,emb_size)
        '''

        batch_size = token_hidden_proj_detach.shape[0]
        seq_len = token_hidden_proj_detach.shape[1]
        mr_pos = estimate_coeff_mat_batch_opt(token_hidden_proj_detach,
                                              facet_emb.transpose(0, 1).detach().clone(), 0.2,
                                              torch.device('cuda'),max_iter=60)
        '''
        Our negative examples are inside in the batch. 
        For example, considering batch size=16 and the first sentences, 
        the first one is positive example and the left are negative examples.
        Write a for loop to select negative examples correspondingly.
        (Need to write in a more efficient way)
        '''
        neg_list = []
        for i in range(batch_size):
            neg_list.append(torch.cat([token_hidden_proj_detach[:i], token_hidden_proj_detach[i + 1:]]).reshape(-1, 768))

        # batch, number_of_tokens*(batch-1)
        neg_list = torch.stack(neg_list)
        mr_neg = estimate_coeff_mat_batch_opt(neg_list,
                                              facet_emb.transpose(0, 1).detach().clone(), 0.2,
                                              torch.device('cuda'))

        dot_pos = torch.sum(torch.bmm(facet_emb.transpose(0, 1).transpose(2, 1),
                                      mr_pos.transpose(2,1)) * token_hidden_proj_detach.transpose(2,1), dim=1)  # -> (Batch, numbers of tokens)
        dot_neg = torch.sum(torch.bmm(facet_emb.transpose(0, 1).transpose(2, 1),
                                      mr_neg.transpose(2, 1)) * neg_list.transpose(2, 1), dim=1)
        prob = []
        '''
        Combine pos and neg, and put pos in the right index for afterward loss computing.
        i.e., the first example should locate at index 0, the second index 1...etc
        '''
        for i in range(batch_size):
            prob.append(torch.cat([dot_neg[i,:i*seq_len], dot_pos[i], dot_neg[i,seq_len*i:]]))
        prob = torch.stack(prob)
        return prob.reshape(batch_size,batch_size,-1) #batch, batch, number_of_tokens

    def est_correlation(self,l,r):
        dot_prod = torch.bmm(l, r.transpose(2,1)).reshape(l.shape[0],-1)

        corr_list = []
        for i in range(l.shape[0]):
            for j in range(i+1,l.shape[0]):
                corr_list.append(self.get_correlation(dot_prod[i],dot_prod[j]))
        return corr_list


    def get_correlation(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cor = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cor
