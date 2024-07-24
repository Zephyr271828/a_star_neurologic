from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = '/scratch/yx3038/tmp/llama3'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def generate_response(prompt, model, tokenizer, max_length=50):

    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    print(inputs)

    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    with open('/scratch/yx3038/Research/StableToolBench/inference_results/constraints/queries.txt', 'r+') as f:
        for idx, prompt in enumerate(f.readlines()):

            print(prompt)

            response = generate_response(prompt, model, tokenizer, max_length = 150)

            print(response)

            break

if __name__ == '__main__':
    main()