import pickle

def make_dict(table_dir):

    dic = {}
    with open(table_dir, 'r+') as f:
        for line in f.readlines():
            words = line.split('\t')
            # print(words)
            en, de = words[2].strip(), words[3].strip()
            dic[de] = en

    return dic

def convert(dic, constraint_dir, output_dir):

    ens = []
    with open(constraint_dir, 'r+') as f:
        for line in f.readlines():
            de = line.strip()[3:-3]
            # print(de)
            en = dic[de]
            ens.append(en)
    
    return ens

def extract(table_dir):
    pass

def test(file_dir):
    max_len = 0
    with open(file_dir, 'r+') as f:
        for line in f.readlines():
            if len(line.strip().split(' ')) > max_len:
                max_len = len(line.strip().split(' '))
    print(max_len)

if __name__ == '__main__':
    constraint_dir = 'dataset/machine_translation/constraint/iate.414.json'
    table_dir = 'dataset/machine_translation/iate/iate.414.terminology.tsv'
    output_dir = 'dataset/machine_translation/iate2.414.json'

    # dic = make_dict(table_dir)
    # print(dic)
    # ens = convert(dic, constraint_dir, output_dir)
    # print(ens)
    test('dataset/commongen/train.txt')