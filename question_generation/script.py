from pprint import pprint

batch_size = 16
sentences = []
with open('slurm/result.out', 'r+') as f:
    for idx, line in enumerate(f.readlines()):
        tmp = line.strip()
        i = 14
        j = tmp[i + 1:].find('score')
        sentence = tmp[i:i + j - 3]
        sentences.append(sentence)
        if idx >= batch_size - 1:
            break

# words = [sentence.split(' ') for sentence in sentences]
# pprint(words)
# pprint(sentences)
# print(len(sentences))
scores = [[] for sentence in sentences]
res = [[] for sentence in sentences]

with open('individual_tokens.txt', 'r+') as f:
    n = 0
    for idx, line in enumerate(f.readlines()):
        tmp = line.strip()
        if tmp != '':
            scores[n].append(tmp)
        else:
            n = (n + 1) % batch_size

        # if idx > 5000:
        #     break

for o, (score, sentence) in enumerate(zip(scores, sentences)):
    i = sentence.find(' ')
    sentence = sentence[i:]
    for each in score:
        k = 2
        l = each.find(', tensor')
        word = each[k : l - 1]
        L = len(word)
        if word == sentence[:L]:
            res[o].append(each)
            # print(each)
            sentence = sentence[L:]

pprint(res)