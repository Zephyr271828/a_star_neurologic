import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = '/scratch/yx3038/tmp/llama3'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# system_message = '''You are a helpful assistant and capable of answering user's queries. Each time you will be given a query and a function response (in a dictionary form), and you need to make use of the information in the function to answer the query. 
#     For instance, you may be given a query: 
#     'I need to retrieve the details of a fake user with a specific gender. Please provide me with the information of a male user.' 
#     and a function response: 
#     '"RESPONSE": {
#         "role": "function",
#         "name": "get_user_by_gender_for_fake_users",
#         "content": "{\"error\": \"\", \"response\": \"{'results': [{'gender': 'male', 'name': {'title': 'Mr', 'first': 'Tanasko', 'last': 'Tasi\\u0107'}, 'location': {'street': {'number': 1439, 'name': 'Protopopa Marka'}, 'city': 'Kraljevo', 'state': 'South Ba\\u010dka', 'country': 'Serbia', 'postcode': 51441, 'coordinates': {'latitude': '57.3087', 'longitude': '107.0263'}, 'timezone': {'offset': '-5:00', 'description': 'Eastern Time (US & Canada), Bogota, Lima'}}, 'email': 'tanasko.tasic@example.com', 'login': {'uuid': 'c42b5d7e-f090-4350-abd8-9a85b877226f', 'username': 'silverswan560', 'password': 'beater', 'salt': '58cJbowc', 'md5': '3659b9c447944e42da3c541739348602', 'sha1': '6ca93a1a1cb5db76fe8c5d1da170c9a52d06f464', 'sha256': '58233f2dcf77b11c348a3d35673aa8e4bc782b07b6fb2d78ba58a5893fc78035'}, 'dob': {'date': '1979-06-26T14:58:36.313Z', 'age': 44}, 'registered': {'date': '2015-02-24T22:18:14.051Z', 'age': 8}, 'phone': '027-5441-149', 'cell': '062-6907-525', 'id': {'name': 'SID', 'value': '746305156'}, 'picture': {'large': 'https://..."
#     }'
#     To answer the query, you should pay attention to the keys in the response, pick the most relevant keys and the corresponding values, and use them as part of your answer. In this case, name, location, email, and timezone are the most relevant keys, so the corresponding values should be included in your answer. In the end, you need to connect these values and answer the query in natural language and nothing else should be included.'''

system_message = "You are a helpful assistant and capable of answering user's queries with a set of powerful functions. Each time you are given a query, you will also have access to the function's response in a dictionary form. You need to make use of the information in the function return to answer the query. Pay attention to the relevant keys and their corresponding values. Articulate them to answer the user's query well."

def generate_response(model, tokenizer, user_message, system_message = None, max_length = 50):

    if system_message is not None:
        inputs = f'{system_message}\n{user_message}'
    else:
        inputs = user_message

    inputs = tokenizer.encode(inputs, return_tensors='pt').to(device)

    n = len(inputs[0])

    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0][n:], skip_special_tokens=True)

    return response

def main():

    idx = 7
    with open(f'/scratch/yx3038/Research/StableToolBench/inference_results/constraints/{idx}.json', 'r+') as f:
        data = json.load(f)

    query = data['QUERY']
    response = str(data['RESPONSE'])
    print('QUERY:\n', query)
    print('RESPONSE:\n', response)

    user_message = 'query:\n' + query + '\nfunction response:\n' + response
    answer = generate_response(
        user_message = user_message, 
        system_message = system_message,
        model = model, 
        tokenizer = tokenizer, 
        max_length = 2048
    )
    print('ANSWER:\n', answer)

    # with open('/scratch/yx3038/Research/StableToolBench/inference_results/constraints/queries.txt', 'r+') as f:
    #     for idx, user_message in enumerate(f.readlines()):

    #         print(user_message)

    #         response = generate_response(
    #             user_message = user_message, 
    #             system_message = None,
    #             model = model, 
    #             tokenizer = tokenizer, 
    #             max_length = 150
    #         )

    #         print(response)

    #         break

if __name__ == '__main__':
    main()