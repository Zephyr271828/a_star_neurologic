# -*- coding: utf-8 -*- 

import os

#os.environ['TRANSFORMERS_CACHE'] = '/scratch/yx3038/cache'
# os.environ['HF_DATASETS_CACHE'] = '/scratch/yx3038/cache'
# os.environ['HF_HOME'] = '/scratch/yx3038/cache'

import json
import math
import argparse
import torch
import torch.nn as nn
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from os import path
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
from transformers import LlamaConfig, GPT2Config

# from flash_attention import FlashAttention

import sys

sys.path.append('..')

from pprint import pprint

from toolLLM.generate import generate
from toolLLM.utils import tokenize_constraints
from toolLLM.lexical_constraints import init_batch


logger = logging.getLogger(__name__)

class FlashAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads):
        super(FlashAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        context_layer = flash_attention(query_layer, key_layer, value_layer, attention_mask)

        output = self.dense(context_layer)
        return output

class GPT2WithFlashAttention(AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.flash_attention = FlashAttention(config.hidden_size, config.num_attention_heads)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):

        embedding_output = self.wte(input_ids)
        
        flash_attention_output = self.flash_attention(embedding_output, attention_mask)
        
        outputs = self.h(flash_attention_output, attention_mask, **kwargs)
        
        return outputs

class LLaMA3WithFlashAttention(AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.flash_attention = FlashAttention(config.hidden_size, config.num_attention_heads)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):

        embedding_output = self.model.embed_tokens(input_ids)
        
        flash_attention_output = self.flash_attention(embedding_output, attention_mask)
        
        outputs = self.model.transformer(flash_attention_output, attention_mask, **kwargs)
        
        return outputs

def main():
    # print(torch.cuda.is_available())

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--input_path", type=str, help="path of input file")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--response_file", type=str, help="response file")
    parser.add_argument("--constraint_file", type=str, help="constraint file")
    parser.add_argument("--key_constraint_file", type=str, help="key elements in constraint file")

    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=10,
                        help="Beam size for searching")
    parser.add_argument('--max_tgt_length', type=int, default=100,
                        help="maximum length of decoded sentences")
    parser.add_argument('--min_tgt_length', type=int, default=0,
                        help="minimum length of decoded sentences")
    parser.add_argument('--ngram_size', type=int, default=3,
                        help='all ngrams can only occur once')
    parser.add_argument('--length_penalty', type=float, default=0.6,
                        help="length penalty for beam search")

    parser.add_argument('--prune_factor', type=int, default=50,
                        help="fraction of candidates to keep based on score")
    parser.add_argument('--sat_tolerance', type=int, default=2,
                        help="minimum satisfied clause of valid candidates")

    # for A star deocding
    parser.add_argument('--look_ahead_step', type=int, default=5,
                        help="number of step to look ahead")
    parser.add_argument('--look_ahead_width', type=int, default=None,
                        help="width of beam in look ahead")
    parser.add_argument('--alpha', type=float, default=0.05,
                        help="decay factor for score in looking ahead")
    parser.add_argument('--fusion_t', type=float, default=None,
                        help="temperature to fuse word embedding for continuous looking ahead")
    parser.add_argument('--look_ahead_sample',  action='store_true',
                        help="whether use sampling for looking ahead")

    args = parser.parse_args()
    pprint(args)

    print(f"Decoding with: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    # model = AutoModelWithLMHead.from_pretrained(args.model_name)

    if torch.cuda.is_available():
        GPU_name = torch.cuda.get_device_name(0)
        if '100' in GPU_name:
            # NOTE bf16 support for model
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
        else:
            # NOTE fp16 support
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
            model = model.half()


    # NOTE add flash attention to the model
    # if 'gpt2' in args.model_name:
    #     # config = GPT2Config.from_pretrained(args.model_name)
    #     # model = GPT2WithFlashAttention.from_config(config)
    #     model = GPT2WithFlashAttention.from_pretrained(args.model_name)
    # elif 'llama3' in args.model_name:
    #     # config = LlamaConfig.from_pretrained(args.model_name)
    #     # model = LLaMA3WithFlashAttention.from_config(config)
    #     model = LLaMA3WithFlashAttention.from_pretrained(args.model_name)
    
    torch.cuda.empty_cache()
    model.eval()
    model = model.to('cuda')
    # no training, only decoding

    PAD_ID = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    eos_ids = [tokenizer.convert_tokens_to_ids(tokenizer.eos_token)]
    # PAD_ID = tokenizer.pad_token_id
    # PAD_ID = 0
    # print('PAD ID:', PAD_ID)

    with open(args.input_path) as fin:
        input_lines = [line for line in fin.read().splitlines()]
    # input lines contain the constraints before "=" only

    def read_constraints(file_name):
        cons_list = []
        with open(file_name, 'r') as f:
            for line in f:
                cons = []
                for concept in json.loads(line):
                    cons.append([f' {c}' for c in concept])
                cons_list.append(cons)
        #print(cons_list)
        return cons_list
    # constraints are in the form of 2d-lists
    # they correspond to the constraints given in input lines

    constraints_list = read_constraints(args.constraint_file)
    key_constraints_list = read_constraints(args.key_constraint_file)
    flattened_list = [[each for l in lst for each in l] for lst in key_constraints_list]
    response_list = json.load(open(args.response_file, 'r+'))
    # both files contain variations of the given word
    # key_constraints_list is case-insenstive, whereas constraints_list is case-sensitive
    # system_message = "You are a helpful assistant and capable of answering user's queries with a list of words. Each time you are given a query, you will also have access to a list of words corresponding to the information required in the query. You need to make use of the list of words to answer the query. Try to figure out which word corresponds to which information required. Articulate them to answer the user's query well."
    system_message = "You are a helpful assistant and capable of answering user's queries with a set of powerful functions. Each time you are given a query, you will also have access to the function's response in a dictionary form. You need to make use of the information in the function return to answer the query. Pay attention to the relevant keys and their corresponding values. Articulate them to answer the user's query well."

    print(len(input_lines))
    input_lines = [
        f'''<|start_header_id|>system<|end_header_id|>
        {system_message}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        query: {input_lines[i]} 
        response: {response_list[i]}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|> ''' for i in range(len(input_lines))]
    # input_lines = [f'query: {each}\nanswer: ' for each in input_lines]
    print(input_lines[0])
    print([len(each) for each in input_lines])

    input_lines = tokenizer(
        input_lines, 
        padding = True, 
        truncation = True, 
        max_length = 256, 
        return_tensors='pt'
    )
    constraints_list = tokenize_constraints(tokenizer, constraints_list)
    key_constraints_list = tokenize_constraints(tokenizer, key_constraints_list)
    # tokenize inputs and constraints

    if path.exists(args.output_file):
        # count = len(open(args.output_file, 'r').readlines())
        fout = Path(args.output_file).open("a", encoding="utf-8")
        # input_lines = input_lines[count:]
        # constraints_list = constraints_list[count:]
        # key_constraints_list = key_constraints_list[count:]
    else:
        fout = Path(args.output_file).open("w", encoding="utf-8")
    total_batch = math.ceil(len(input_lines['input_ids']) / args.batch_size)
    next_i = 0
    # output file directory
    # if output already exists, append, otherwise write

    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines['input_ids']):
        # iterate over the batches
            # _chunk = [[tokenizer.bos_token_id for j in range(len(each))] for each in _chunk]
            constraints = init_batch(raw_constraints=constraints_list[next_i:next_i + args.batch_size],
                                     key_constraints=key_constraints_list[next_i:next_i + args.batch_size],
                                     beam_size=args.beam_size,
                                     eos_id=eos_ids)
            # init_batch from lexical constraints
            # effects: 

            input_ids = input_lines['input_ids'][next_i:next_i + args.batch_size, :]
            input_ids = input_ids.to('cuda')
            attention_mask = input_lines['attention_mask'][next_i:next_i + args.batch_size, :]
            attention_mask = attention_mask.to('cuda')
            # print(input_ids.shape)
            # print(attention_mask.shape)

            outputs = generate(self=model,
                               input_ids=input_ids,
                               attention_mask=attention_mask,
                               pad_token_id=PAD_ID,
                               min_length=args.min_tgt_length,
                               max_length=args.max_tgt_length,
                               num_beams=args.beam_size,
                               no_repeat_ngram_size=args.ngram_size,
                               length_penalty=args.length_penalty,
                               constraints=constraints,
                               prune_factor=args.prune_factor,
                               sat_tolerance=args.sat_tolerance,
                               look_ahead_step=args.look_ahead_step,
                               look_ahead_width=args.look_ahead_width,
                               alpha=args.alpha,
                               fusion_t=args.fusion_t,
                               look_ahead_sample=args.look_ahead_sample)
            # generate from commogen.generate
            # effects:
            
            output_sequences = [tokenizer.decode(o) for i, o in enumerate(outputs)]
            # after checking the output sequences, somehow only the last prompt in a batch has a corresponding sentence as output
            # while the last sentence has a weird result, with a bunch of numbers
            # need to check generate()

            print(len(output_sequences))
            print(output_sequences)
            for hypothesis in output_sequences:
                tmp = hypothesis.strip().replace('<|endoftext|>', '')
                print(tmp)
                fout.write(tmp + "\n")
                fout.flush()
                # clear the buffer of a file

            pbar.update(1)
            next_i += args.batch_size

            break

if __name__ == "__main__":
    main()
