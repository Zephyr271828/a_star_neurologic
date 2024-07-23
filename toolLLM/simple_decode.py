import pdb
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = 'fine-tuned_gpt2/checkpoint-14500'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_path)
model.generation_config.pad_token_ids = tokenizer.pad_token_id

def parse_input(input_dir):
    input_ids = []

    with open(input_dir, 'r+') as f:
        input_ids = tokenizer([line.split('=')[0] + '=' for line in f.readlines()])

    return input_ids

def generate(input_ids, output_dir):
    with open(output_dir, 'w+') as f:
        pass
    input_ids = [torch.tensor([v]) for v in input_ids['input_ids']]
    print(input_ids[0])
    with open(output_dir, 'a+') as f:
        for idx, each in enumerate(tqdm(input_ids)):
            out = model.generate(each)
            out = [tokenizer.decode(o) for o in out]
            if idx == 0:
                print(out)
            f.write(out[0])
            f.write('\n')
    # print(out)

if __name__ == '__main__':
    input_dir = '../dataset/commongen/dev.txt'
    output_dir = 'output.txt'

    input_ids = parse_input(input_dir)
    generate(input_ids, output_dir)