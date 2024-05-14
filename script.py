import json

def load_constraints(file_path):
    constraints = []

    with open(file_path, 'r+') as f:
        for line in f.readlines():
            data = json.loads(line)
            constraints.append(data)
    
    return constraints

def check_clause(clause, sentence):
    for vocab in clause:
        if vocab in sentence:
            return True

    return False

def check_outputs(file_path, constraints, beam_size = 12):
    tot = 0
    correct = 0

    with open('anomaly.txt', 'w+') as f_a:
        f_a.write('')

    with open(file_path, 'r+') as f:
        for idx,line in enumerate(f.readlines()):
            try:
                data = json.loads(line)
                sentence = data['sentence']
            except:
                sentence = line

            i = idx // beam_size
            for clause in constraints[i]:
                if not check_clause(clause, sentence):
                    with open('anomaly.txt', 'a+') as f_a:
                        f_a.write('{}\n'.format(constraints[i]))
                        f_a.write(line)
                else:
                    correct += 1
                tot += 1
    return correct / tot

if __name__ == '__main__':
    config = {
        'cmg_u' : {
            'constraint_path' : 'dataset/commongen/constraint/dev.constraint.json',
            'output_path' : 'commongen_unsupervised/slurm/result.out',
            'beam_size' : 12
        },
        'translation' : {
            'constraint_path' : 'dataset/machine_translation/constraint/iate.414.json',
            'output_path' : 'translation/slurm/result.out',
            'beam_size' : 1
        },
        'question' : {
            'constraint_path' : 'dataset/question_generation/constraints.jsonl',
            'output_path' : 'question_generation/slurm/result.out',
            'beam_size' : 9
        },
        'question2' : {
            'constraint_path' : 'dataset/question_generation/constraints.jsonl',
            'output_path' : 'question_generation/slurm/result3.out',
            'beam_size' : 9
        }
    }
    d = config['cmg_u']

    constraints = load_constraints(d['constraint_path'])
    coverage = check_outputs(d['output_path'], constraints, d['beam_size'])
    print('constraint coverage: {}'.format(coverage))