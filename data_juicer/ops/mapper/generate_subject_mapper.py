import os
from time import time
import torch
# To use one neuron core per worker
from datasets import load_dataset
import numpy as np
from transformers import Qwen2Model, Qwen2ForSequenceClassification,AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import Dataset
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    SequenceClassifierOutputWithPast
    )
from typing import List,Optional, Tuple, Union


import json
import random
import re
from typing import Dict, Optional
from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, UNFORKABLE, Mapper


 
class Qwen2ForSequenceClassificationNewV1(Qwen2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        # self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.config.problem_type = "single_label_classification"
        self.num_labels = 10#config.num_labels
        self.num_labels_h2 = 4#config.num_labels_h2
        self.num_labels_h3 = 2#config.num_labels_h2
        self.num_labels_h4 = 4#config.num_labels_h2        
        self.num_labels_h5 = 5#config.num_labels_h2        
        
        classifier_dropout = 0.1
        '''
        #head 1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        #head 2
        self.dropout_2 = nn.Dropout(classifier_dropout)
        self.classifier_2 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),            
            nn.Linear(config.hidden_size, self.num_labels_h2)
        )
        #head 3
        self.dropout_3 = nn.Dropout(classifier_dropout)
        self.classifier_3 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),              
            nn.Linear(config.hidden_size, self.num_labels_h3)   
        )

        #head 4
        self.dropout_4 = nn.Dropout(classifier_dropout)
        self.classifier_4 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),              
            nn.Linear(config.hidden_size, self.num_labels_h4)   
        )
        '''
        #head 1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.LayerNorm(config.hidden_size*2),
            nn.SiLU(),
            nn.Linear(config.hidden_size*2, config.hidden_size)
        )
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),                    
            nn.Linear(config.hidden_size, self.num_labels),
            nn.LayerNorm(self.num_labels),

        )

        #head 2
        self.dropout_2 = nn.Dropout(classifier_dropout)
        self.classifier_2 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.LayerNorm(config.hidden_size*2),
            nn.SiLU(),
            nn.Linear(config.hidden_size*2, config.hidden_size)
        )
        self.classifier_2_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),                    
            nn.Linear(config.hidden_size, self.num_labels_h2),
            nn.LayerNorm(self.num_labels_h2),
        )              

        #head 3
        self.dropout_3 = nn.Dropout(classifier_dropout)
        self.classifier_3 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.LayerNorm(config.hidden_size*2),
            nn.SiLU(),
            nn.Linear(config.hidden_size*2, config.hidden_size)
        )
        self.classifier_3_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),                    
            nn.Linear(config.hidden_size, self.num_labels_h3),
            nn.LayerNorm(self.num_labels_h3),

        )              

        #head 4
        self.dropout_4 = nn.Dropout(classifier_dropout)
        self.classifier_4 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.LayerNorm(config.hidden_size*2),
            nn.SiLU(),
            nn.Linear(config.hidden_size*2, config.hidden_size)
        )
        self.classifier_4_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),                    
            nn.Linear(config.hidden_size, self.num_labels_h4),
            nn.LayerNorm(self.num_labels_h4),

        )        
        #head 5
        self.dropout_5 = nn.Dropout(classifier_dropout)
        self.classifier_5 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            nn.LayerNorm(config.hidden_size*2),
            nn.SiLU(),
            nn.Linear(config.hidden_size*2, config.hidden_size)
        )
        self.classifier_5_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),                    
            nn.Linear(config.hidden_size, self.num_labels_h5),
            nn.LayerNorm(self.num_labels_h5),

        )      
        # Initialize weights and apply final processing
        self.post_init()

        # #need init 
        # for m_ in [self.classifier,self.classifier_head,self.classifier_2,self.classifier_2_head,self.classifier_3,self.classifier_3_head,self.classifier_4,self.classifier_4_head]:
        #     for m_i in m_:
        #         _init_weights(m_i)      
          
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        pooled_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        # logits = self.score(pooled_hidden_states)
        pooled_output = pooled_hidden_states
        '''
        #head 1
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        #head 2
        pooled_output_2 = self.dropout_2(pooled_output)
        logits_2 = self.classifier_2(pooled_output_2)

        #head 3
        pooled_output_3 = self.dropout_3(pooled_output)
        logits_3 = self.classifier_3(pooled_output_3)

        #head 4
        pooled_output_4 = self.dropout_4(pooled_output)
        logits_4 = self.classifier_4(pooled_output_4)   
        '''
        #head 1
        logits_out = self.classifier(pooled_output)
        pooled_output_1 = self.dropout(logits_out+pooled_output)
        logits = self.classifier_head(pooled_output_1)

        #head 2
        logits_2_out = self.classifier_2(pooled_output)
        pooled_output_2 = self.dropout_2(logits_2_out+pooled_output)
        logits_2 = self.classifier_2_head(pooled_output_2)

        #head 3
        logits_3_out = self.classifier_3(pooled_output)
        pooled_output_3 = self.dropout_3(logits_3_out+pooled_output)
        logits_3 = self.classifier_3_head(pooled_output_3)

        #head 4
        logits_4_out = self.classifier_4(pooled_output)     
        pooled_output_4 = self.dropout_4(logits_4_out+pooled_output)
        logits_4 = self.classifier_4_head(pooled_output_4)

        #head 5
        logits_5_out = self.classifier_5(pooled_output)     
        pooled_output_5 = self.dropout_5(logits_5_out+pooled_output)
        logits_5 = self.classifier_5_head(pooled_output_5)


        loss = None
        if labels is not None:
            labels = labels.to(hidden_states.device)
            # if self.config.problem_type is None:
            #     if self.num_labels == 1:
            #         self.config.problem_type = "regression"
            #     elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            #         self.config.problem_type = "single_label_classification"
            #     else:
            #         self.config.problem_type = "multi_label_classification"

            # if self.config.problem_type == "regression":
            #     loss_fct = MSELoss()
            #     if self.num_labels == 1:
            #         loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            #     else:
            #         loss = loss_fct(pooled_logits, labels)
            # elif self.config.problem_type == "single_label_classification":
            #     loss_fct = CrossEntropyLoss()
            #     loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            # elif self.config.problem_type == "multi_label_classification":
            #     loss_fct = BCEWithLogitsLoss()
            #     loss = loss_fct(pooled_logits, labels)
            '''
            loss_fct = CrossEntropyLoss()
            head_1_label = labels[:,0]
            loss_1 = loss_fct(logits[head_1_label!=-1].view(-1, self.num_labels), labels[:,0][head_1_label!=-1].view(-1))

            head_2_label = labels[:,1]
            loss_2 = loss_fct(logits_2[head_2_label!=-1].view(-1, self.num_labels_h2), labels[:,1][head_2_label!=-1].view(-1))

            head_3_label = labels[:,2]
            loss_3 = loss_fct(logits_3[head_3_label!=-1].view(-1, self.num_labels_h3), labels[:,2][head_3_label!=-1].view(-1))

            head_4_label = labels[:,3]
            loss_4 = loss_fct(logits_4[head_4_label!=-1].view(-1, self.num_labels_h4), labels[:,3][head_4_label!=-1].view(-1))
            
            loss = loss_1+loss_2+loss_3+loss_4
            '''
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            ''' head1 '''
            head_1_label = labels[:,0]
            loss_1 = loss_fct(logits.view(-1, self.num_labels), labels[:,0].view(-1))
            if  torch.isnan(loss_1):
                loss_1 = torch.zeros_like(loss_1, requires_grad=True)#torch.tensor(0, dtype=torch.float32)#0
            # loss = loss_1
            
            ''' head2 '''
            head_2_label = labels[:,1]
            loss_2 = loss_fct(logits_2.view(-1, self.num_labels_h2), labels[:,1].view(-1))
            if  torch.isnan(loss_2):
                loss_2 = torch.zeros_like(loss_2, requires_grad=True)#torch.tensor(0, dtype=torch.float32)#0                
            # loss = loss_2
            
            head_3_label = labels[:,2]
            loss_3 = loss_fct(logits_3.view(-1, self.num_labels_h3), labels[:,2].view(-1))
            if  torch.isnan(loss_3):
                loss_3 = torch.zeros_like(loss_3, requires_grad=True)#torch.tensor(0, dtype=torch.float32)#0                
            # loss = loss_3
            
            ''' head4'''
            head_4_label = labels[:,3]
            loss_4 = loss_fct(logits_4.view(-1, self.num_labels_h4), labels[:,3].view(-1))
            if  torch.isnan(loss_4):
                loss_4 = torch.zeros_like(loss_4, requires_grad=True)#torch.tensor(0, dtype=torch.float32)#0

            ''' head5'''
            head_5_label = labels[:,3]
            loss_5 = loss_fct(logits_5.view(-1, self.num_labels_h5), labels[:,4].view(-1))
            if  torch.isnan(loss_5):
                loss_5 = torch.zeros_like(loss_5, requires_grad=True)#torch.tensor(0, dtype=torch.float32)#0

            loss = loss_1+loss_2+loss_3+loss_4+loss_5
            if loss == 0:
                print('nan:loss=0')

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=(logits,logits_2,logits_3,logits_4,logits_5),
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


head1_map={
    0:'语文',
    1:'数学',
    2:'英语',
    3:'物理',
    4:'化学',
    5:'生物',
    6:'历史',
    7:'地理',
    8:'政治',
    9:'其他',
    -1:'不确定'
}

head2_map={
    0:'多轮',
    1:'单轮-其他',
    2:'单轮-数值',
    3:'单轮-符号',
    -1:'不确定'
}
head3_map={
    0:'主观',
    1:'客观',
    -1:'不确定'
}
head4_map={
    0:'小学',
    1:'初中',
    2:'高中',
    3:'其他',
    -1:'不确定'
}
head5_map={
    0:'其他',
    1:'数值',
    2:'符号',
    3:'作文',
    4:'结构化',
    -1:'不确定'
}

OP_NAME = 'generate_subject_mapper'

with AvailabilityChecking(['torch', 'transformers', 'vllm'], OP_NAME):
    import torch
    import transformers  # noqa: F401
    import vllm  # noqa: F401

    # avoid hanging when calling model in multiprocessing
    torch.set_num_threads(1)


# TODO: Extend LLM-based OPs into API-based implementation.
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class GenerateSubjectMapper(Mapper):
    """Mapper to generate subject for all data.
    You should configure an empty dataset in your yaml config file:
    ```
    generated_dataset_config:
      type: 'EmptyFormatter'  # use `RayEmptyFormatter` when enable ray
      length: ${The number of generated samples}
      feature_keys: ${text key}
    ```
    The number of samples generated is determined by
    the length of the empty dataset.
    """
    _accelerator = 'cuda'

    def __init__(self,
                 hf_model: str = '/mnt/pfs/jinfeng_team/IT/gwxie/modelFiles/qwen2/train_3_head/train_qwen2_4_head_2048token_pretrain_v2_3/checkpoint-1242',
                 enable_vllm: bool = True,
                 tensor_parallel_size: Optional[int] = None,
                 max_num_seqs: int = 256,
                 max_input_length: int=512,
                 *args,
                 **kwargs):
        """
        Initialization method.
        :param hf_model: Hugginface model id.     
        :param enable_vllm: Whether to use vllm for inference acceleration.
        :param tensor_parallel_size: It is only valid when enable_vllm is True.
            The number of GPUs to use for distributed execution with tensor
            parallelism.
        :param max_model_len: It is only valid when enable_vllm is True.
            Model context length. If unspecified, will be automatically
            derived from the model config.
        :param max_num_seqs: It is only valid when enable_vllm is True.
            Maximum number of sequences to be processed in a single iteration.
        :param sampling_params: Sampling parameters for text generation.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.num_proc = 1
        self.enable_vllm = enable_vllm
        self.max_input_length=max_input_length
        self.model_tokenizer_model_config = self.model_fn_change_max_length(hf_model,max_input_length)

        # if enable_vllm:
        #     import torch
        #     from vllm import SamplingParams

        #     assert torch.cuda.device_count() >= 1, 'must be executed in CUDA'
        #     if not tensor_parallel_size:
        #         tensor_parallel_size = torch.cuda.device_count()
        #         logger.info(f'Set tensor_parallel_size to \
        #             {tensor_parallel_size} for vllm.')
        #     self.model_key = prepare_model(
        #         model_type='vllm',
        #         pretrained_model_name_or_path=hf_model,
        #         trust_remote_code=trust_remote_code,
        #         tensor_parallel_size=tensor_parallel_size,
        #         max_model_len=max_model_len,
        #         max_num_seqs=max_num_seqs)
        #     self.sampling_params = SamplingParams(**sampling_params)
        # else:
        #     self.model_key = prepare_model(
        #         model_type='huggingface',
        #         pretrained_model_name_or_path=hf_model,
        #         trust_remote_code=trust_remote_code)
        #     self.sampling_params = sampling_params
        #     print("sampling_params: ",self.sampling_params) 
            
        # load all samples data for generate tag
        # self.data_samples = self.load_qa_samples(data_file)

        # if len(self.data_samples) == 0:
            # raise ValueError('No QA data was parsed from the seed file!')


    def model_fn_change_max_length(self,model_dir, max_input_length=512):
        # load tokenizer and neuron model from model_dir
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.truncation_side='left'
        tokenizer.model_max_length = max_input_length
        # tokenizer = AutoTokenizer.from_pretrained('/tal-vePFS/RM/gwxie/preTrain/huggingface/BERT/bert-base-chinese')
        # model = torch.jit.load(os.path.join(model_dir, AWS_NEURON_TRACED_WEIGHTS_NAME))
        model_config = AutoConfig.from_pretrained(model_dir)
        print("count: ",torch.cuda.device_count())
        model = Qwen2ForSequenceClassificationNewV1.from_pretrained(model_dir).half().cuda()
        model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer, model_config


    def gen_input(self,sample):
        txt_ = sample[self.text_key]
        txt_=sample['question']
        id_=sample['question_id']
        txt_list = txt_.split('[SEP]')
        for txt_n, txt_i in enumerate(txt_list):
            if txt_n % 2 == 0 and txt_n!=0:
                txt_list[txt_n] = '[SEP]'+txt_i
            elif txt_n % 2 == 1 and txt_n!=0:
                txt_list[txt_n] = '<r>'+txt_i
        txt_ = ''.join(txt_list)
        # label
        # 学科[0~9]、单多轮[0-1]、主客[0-1]、学段[0-3]、助手[0-4]      
        cls_ = [-1,-1,-1,-1,-1]     
        # input data
        d_ = {
                'id':id_,
                'label':cls_,
                'text':txt_
        }
        return d_   


    # predict
    def predict_fn_5class(self,data, model_tokenizer_model_config,max_length=2048,save_path=None):
        # destruct model, tokenizer and model config
        model, tokenizer, model_config = model_tokenizer_model_config
        result = data.copy()
        # create embeddings for inputs
        inputs = data.pop("text", data)
        #inputs = '竖式计算：2+2=<r>2+2=<span>4,0=4<\\span>\n答案是：4[SEP]竖式计算：2+2='
        special_token = '[SEP]'

        if special_token in inputs:
            txt_l = inputs.split(special_token)
            txt_1 = special_token.join(txt_l[:-1])
            txt_2 = txt_l[-1]
        else:
            txt_1 = inputs
            txt_2 = ''    
        # embeddings = tokenizer(
        #     txt_1,txt_2,
        #     return_tensors="pt",
        #     max_length=model_config.max_position_embeddings,
        #     padding="max_length",
        #     truncation=True,
        # )
        embeddings = tokenizer(inputs,  
                                max_length=max_length,
                                truncation=True,
                                return_tensors="pt")            
                                # padding='max_length', 
        # convert to tuple for neuron model
        # neuron_inputs = tuple(embeddings.values())
        embeddings = {k: v.cuda() for k, v in embeddings.items()}

        # run prediciton
        with torch.no_grad():
            # predictions = model(*neuron_inputs)[0]
            # predictions = model(**embeddings)[0]
            predictions = model(**embeddings)#[0]
            heads = predictions.logits
            heads_1 = torch.nn.Softmax(dim=1)(heads[0])
            heads_2 = torch.nn.Softmax(dim=1)(heads[1])
            heads_3 = torch.nn.Softmax(dim=1)(heads[2])
            heads_4 = torch.nn.Softmax(dim=1)(heads[3])
            heads_5 = torch.nn.Softmax(dim=1)(heads[4])

            heads_1_class = heads_1.argmax()
            heads_2_class = heads_2.argmax()
            heads_3_class = heads_3.argmax()
            heads_4_class = heads_4.argmax()#[:,:-1]
            heads_5_class = heads_5.argmax()
        '''
        # if [heads_1_class.item(),heads_2_class.item(),heads_3_class.item(),heads_4_class.item()] != data['label']:
        if heads_1_class.item() != data['label'][0]:
            # pre_ = (heads_1_class,heads_2_class)
            print(inputs, (xueke_map[heads_1_class.item()]+'-'+xueke_map[data['label'][0]], head2_map[heads_2_class.item()]+'-'+head2_map[data['label'][1]], head3_map[heads_3_class.item()]+'-'+head3_map[data['label'][2]], head4_map[heads_4_class.item()]+'-'+head4_map[data['label'][3]]))
            print()
        '''    
        '''
        if heads_1_class.item() != data['label'][0]:
            print(inputs, (xueke_map[heads_1_class.item()]+'     '+xueke_map[data['label'][0]]))
            print()
            with open('/tal-vePFS/RM/gwxie/dataset/trash/test/all_xueke_6-19.json', 'a')as f:
                f.write(json.dumps({'id':data['id'],'prompt':inputs, 'pre_label':xueke_map[heads_1_class.item()],'label':xueke_map[data['label'][0]]}, ensure_ascii=False)+'\n')
        '''    
        '''
        if heads_2_class.item() != data['label'][1]:
            # pre_ = (heads_1_class,heads_2_class)
            print(inputs, (head2_map[heads_2_class.item()]+'     '+head2_map[data['label'][1]]))
            print()
        '''
        '''
        if heads_3_class.item() != data['label'][2]:
            # pre_ = (heads_1_class,heads_2_class)
            print(inputs, (head3_map[heads_3_class.item()]+'     '+head3_map[data['label'][2]]))
            print()
        '''    
        '''
        if heads_4_class.item() != data['label'][3]:
            # pre_ = (heads_1_class,heads_2_class)
            print(inputs, (head4_map[heads_4_class.item()]+'     '+head4_map[data['label'][3]]))
            print()    
        '''

        '''
        if (data['label'][0]!=-1 and data['label'][0]!=heads_1_class.item()) or (data['label'][1]!=-1 and data['label'][1]!=heads_2_class.item()) or (data['label'][2]!=-1 and data['label'][2]!=heads_3_class.item()) or (data['label'][3]!=-1 and data['label'][3]!=heads_4_class.item()):
            with open('/tal-vePFS/RM/gwxie/dataset/trash/test/all_3-28.json', 'a')as f:
                pre_label_ = (xueke_map[heads_1_class.item()],head2_map[heads_2_class.item()],head3_map[heads_3_class.item()],head4_map[heads_4_class.item()])
                label_ = (xueke_map[data['label'][0]],head2_map[data['label'][1]],head3_map[data['label'][2]],head4_map[data['label'][3]])
                f.write(json.dumps({'id':data['id'],'prompt':inputs, 'pre_label':pre_label_,'label':label_}, ensure_ascii=False)+'\n')
        '''

        return (heads_1_class.item(),heads_2_class.item(),heads_3_class.item(),heads_4_class.item(),heads_5_class.item()), data['label'] if 'label' in data else ''



    def process(self, sample=None, rank=None):
        if not sample:
            raise ValueError("sample is None")
        print("sample: ",sample)
        # gen input
        input_d=self.gen_input(sample)
        pre, label = self.predict_fn_5class(input_d, self.model_tokenizer_model_config,max_length=self.max_input_length)#multi head
        subject=head1_map[pre[0]]
        if not subject:
            return {self.text_key: json.dumps({'subject': subject})}
        return {
            self.text_key:
            json.dumps({'subject': subject}, ensure_ascii=False)
        }
