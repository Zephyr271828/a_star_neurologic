import json
import math
import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from os import path
from itertools import islice
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM

from pprint import pprint

import sys
sys.path.append('..')

from toolLLM2.init_beam import get_init_candidate
from toolLLM.generate import generate
from toolLLM.utils import tokenize_constraints
from toolLLM.lexical_constraints import init_batch

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="pretrained language model to use")
    parser.add_argument("--input_path", type=str, help="path of input file")
    parser.add_argument("--output_file", type=str, help="output file")
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
    # model = AutoModelWithLMHead.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    torch.cuda.empty_cache()
    model.eval()
    model = model.to('cuda')

    period_id = [tokenizer.convert_tokens_to_ids('.')]
    period_id.append(tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [tokenizer.eos_token_id] + period_id
    PAD_ID = tokenizer.convert_tokens_to_ids('<pad>')
    bad_token = [':', "'", '-', '_', '@', 'Ċ', 'Ġ:', 'Ġwho', "'s"]
    bad_words_ids = [tokenizer.convert_tokens_to_ids([t]) for t in bad_token]

    def read_constraints(file_name):
        cons_list = []
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                cons = []
                for concept in json.loads(line):
                    cons.append([f' {c}' for c in concept if c.islower()])
                cons_list.append(cons)
        return cons_list
    # return the constraints in the form of a 3-d list

    constraints_list = read_constraints(args.constraint_file)
    key_constraints_list = read_constraints(args.key_constraint_file)
    # given c.islower() is included in the read_constraints() function, I feel the key_constraint_file alone is sufficient

    # beam_inits = get_init_candidate(constraints_list, beam_size=args.beam_size, add_space=False)
    beam_inits = [[l] for l in open(args.input_path, 'r+').readlines()]
    print(beam_inits[0])
    # Add "the""a""an" to all the role-like constraints

    init_factor = [len(x) for x in beam_inits]

    input_lines = [y for x in beam_inits for y in x]
    # pprint(input_lines)
    # input_lines = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in input_lines]
    input_lines = tokenizer(
        input_lines, 
        padding = True, 
        truncation = True, 
        max_length = 77, 
        return_tensors='pt'
    )

    def expand_factor(items, factors):
        expanded_items = []
        for item, factor in zip(items, factors):
            expanded_items.extend([item] * factor)
        return expanded_items

    constraints_list = tokenize_constraints(tokenizer, constraints_list)
    key_constraints_list = tokenize_constraints(tokenizer, key_constraints_list)
    constraints_list = expand_factor(constraints_list, init_factor)
    key_constraints_list = expand_factor(key_constraints_list, init_factor)

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

    logs = []
    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            constraints = init_batch(raw_constraints=constraints_list[next_i:next_i + args.batch_size],
                                     key_constraints=key_constraints_list[next_i:next_i + args.batch_size],
                                     beam_size=args.beam_size,
                                     eos_id=eos_ids)
            # buf = _chunk
            next_i += args.batch_size

            # buf = [x + [PAD_ID] * (max_len - len(x)) for x in buf]

            input_ids = input_lines['input_ids'][next_i:next_i + args.batch_size, :]
            _chunk = input_ids
            input_ids = input_ids.to('cuda')
            attention_mask = input_lines['attention_mask'][next_i:next_i + args.batch_size, :]
            attention_mask = attention_mask.to('cuda')

            advanced_constraints = []
            for j, init_cons in enumerate(constraints):
                adv_cons = init_cons
                for token in _chunk[j // args.beam_size]:
                    adv_cons = adv_cons.advance(token)
                advanced_constraints.append(adv_cons)

            outputs, scores, sum_logprobs = generate(self=model,
                                                     input_ids=input_ids,
                                                     attention_mask=attention_mask,
                                                     pad_token_id=PAD_ID,
                                                     bad_words_ids=bad_words_ids,
                                                     min_length=args.min_tgt_length,
                                                     max_length=args.max_tgt_length,
                                                     num_beams=args.beam_size,
                                                     no_repeat_ngram_size=args.ngram_size,
                                                     length_penalty=args.length_penalty,
                                                     constraints=advanced_constraints,
                                                     prune_factor=args.prune_factor,
                                                     sat_tolerance=args.sat_tolerance,
                                                     look_ahead_step=args.look_ahead_step,
                                                     look_ahead_width=args.look_ahead_width,
                                                     alpha=args.alpha,
                                                     fusion_t=args.fusion_t,
                                                     look_ahead_sample=args.look_ahead_sample)

            output_sequences = [tokenizer.decode(o) for i, o in enumerate(outputs)]

            for hypothesis, score, sum_logprob in zip(output_sequences, scores, sum_logprobs):
                log = json.dumps({'sentence': hypothesis.strip().replace('<|endoftext|>', ''),
                                  'score': score, 'sum_logprob': sum_logprob})
                logs.append(log)
                fout.write(f'{log}\n')
                fout.flush()

            pbar.update(1)

    logs_iter = iter(logs)
    logs_list = [list(islice(logs_iter, elem)) for elem in init_factor]
    logs_list = [sorted([json.loads(s) for s in log_list], key=lambda x: x['score'], reverse=True)[0]
                 for log_list in logs_list]
    selected_outputs = [x['sentence'] for x in logs_list]

    with open(f'{args.output_file}.select', 'w') as f:
        for sentence in selected_outputs:
            f.write(f'{sentence.strip()}\n')


if __name__ == "__main__":
    main()
