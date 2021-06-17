# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for wrapping BertModel."""

import torch

from olfmlm.model.modeling import BertConfig
from olfmlm.model.modeling import BertLayerNorm
from olfmlm.model.new_models import BertHeadTransform

from olfmlm.model.new_models import Bert

def get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0}
    for module_ in module.modules():
        if isinstance(module_, (BertLayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])
    return weight_decay_params, no_weight_decay_params


def get_params_for_dict_format(module_dict):
    '''
    Although the above function also can return the same result, just create another one in case of other issues.
    '''
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0}
    for module_obj in module_dict:
        if isinstance(module_obj, BertHeadTransform):
            continue
        for module_ in module_obj.modules():
            if isinstance(module_, (BertLayerNorm, torch.nn.LayerNorm)):
                no_weight_decay_params['params'].extend(
                    [p for p in list(module_._parameters.values())
                     if p is not None])
            else:
                weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                     if p is not None and n != 'bias'])
                no_weight_decay_params['params'].extend(
                    [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])
    return weight_decay_params, no_weight_decay_params


class BertModel(torch.nn.Module):

    def __init__(self, tokenizer, args):

        super(BertModel, self).__init__()
        # if args.pretrained_bert:
        #     self.model = BertForPreTraining.from_pretrained(
        #         args.tokenizer_model_type,
        #         cache_dir=args.cache_dir,
        #         fp32_layernorm=args.fp32_layernorm,
        #         fp32_embedding=args.fp32_embedding,
        #         layernorm_epsilon=args.layernorm_epsilon)
        # else:
        if args.bert_config_file is None:
            raise ValueError("If not using a pretrained_bert, please specify a bert config file")
        self.config = BertConfig(args.bert_config_file)
        model_args = [self.config]
        # if self.model_type == "referential_game":
        #     self.small_config = BertConfig(args.bert_small_config_file)
        #     model_args.append(self.small_config)


        self.model = Bert(*model_args, modes=args.modes.split(','), extra_token=args.extra_token,
                          agg_function=args.agg_function, unnorm_facet=args.unnorm_facet,
                          unnorm_token=args.unnorm_token, facet2facet=args.facet2facet,
                          use_dropout=args.use_dropout, autoenc_reg_const=args.autoenc_reg_const,
                          use_double=args.double)
        if args.pretrained_bert:
            print('use pretrained weight')
            self.model.bert=self.model.bert.from_pretrained('bert-base-uncased',cache_dir=args.cache_dir,config_file_path=args.bert_config_file)
            if 'mf' in args.modes:
                with torch.no_grad():
                    self.model.lm.decoder.weight=self.model.bert.embeddings.word_embeddings.weight
                    if args.same_weight:
                        self.model.sent.mf.v_1.dense.weight=torch.nn.Parameter(self.model.bert.pooler.dense.weight.data)
                        self.model.sent.mf.v_2.dense.weight=torch.nn.Parameter(self.model.sent.mf.v_1.dense.weight.data)
                        self.model.sent.mf.v_3.dense.weight=torch.nn.Parameter(self.model.sent.mf.v_1.dense.weight.data)
                        self.model.sent.mf.s_1.dense.weight=torch.nn.Parameter(self.model.bert.pooler.dense.weight.data)
                        self.model.sent.mf.s_2.dense.weight=torch.nn.Parameter(self.model.sent.mf.s_1.dense.weight.data)
                        self.model.sent.mf.s_3.dense.weight=torch.nn.Parameter(self.model.sent.mf.s_1.dense.weight.data)

            #self.model.bert=self.model.bert.from_pretrained('bert-base-uncased',cache_dir=args.cache_dir,config_file_path=args.bert_config_file)
            
            

    def forward(self, modes, input_tokens, token_type_ids=None, task_ids=None, attention_mask=None, checkpoint_activations=False, first_pass=False):
        return self.model(modes, input_tokens, token_type_ids, task_ids, attention_mask, checkpoint_activations=checkpoint_activations)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix,
                                     keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def get_params(self):
        param_groups = []
        param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.encoder.layer))
        param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.pooler))
        param_groups += list(get_params_for_weight_decay_optimization(self.model.bert.embeddings))
        for classifier in self.model.sent.values():
            if isinstance(classifier, torch.nn.ModuleDict):
                #handle the mf dict type
                classifier = classifier.values()
                param_groups += list(get_params_for_dict_format(classifier))
            else:
                param_groups += list(get_params_for_weight_decay_optimization(classifier))
        for k, classifier in self.model.tok.items():
            if k == "sbo":
                param_groups += list(get_params_for_weight_decay_optimization(classifier.transform))
                param_groups[1]['params'].append(classifier.bias)
            else:
                param_groups += list(get_params_for_weight_decay_optimization(classifier))
        param_groups += list(get_params_for_weight_decay_optimization(self.model.lm.transform))
        param_groups[1]['params'].append(self.model.lm.bias)

        return param_groups
