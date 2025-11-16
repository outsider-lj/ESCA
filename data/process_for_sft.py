import json
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import nltk
import random
from collections import Counter
import argparse

# random.seed(13)


def str2bool(x):
    if x == "True":
        return True
    elif x == "False":
        return False
    else:
        raise argparse.ArgumentTypeError("must be True or False")


parser = argparse.ArgumentParser()
parser.add_argument('--add_persona', type=str2bool, default=True,
                    help="True or False, this determine whether to use ESConv with persona")
args = parser.parse_args()


def _norm(x):
    return ' '.join(x.strip().split())


# strategies = json.load(open('./strategy.json'))
# strategies = [e[1:-1] for e in strategies]
# strat2id = {strat: i for i, strat in enumerate(strategies)}
# print(args.add_persona)
original = json.load(open(''))


def process_data(d):
    emotion = d['emotion_type']
    problem = d["problem_type"]
    situation = d['situation']
    index=d['index']
    # persona = d['persona']
    # persona_list = d['persona_list']
    # persona_list = []
    # init_intensity = int(d['score']['speaker']['begin_intensity'])
    # final_intensity = int(d['score']['speaker']['end_intensity'])

    d = d['dialog']

    res = {
        'index':index,
        'emotion_type': emotion,
        'problem_type': problem,
        'situation': situation,
        # 'init_intensity': init_intensity,
        # 'final_intensity': final_intensity,
        'dialog': d,
    }
    return res


data = []


for e in tqdm(original, total=len(original)):
    data.append(process_data(e))

emotions = Counter([e['emotion_type'] for e in data])
problems = Counter([e['problem_type'] for e in data])
print('emotion', emotions)
print('problem', problems)

# random.shuffle(data)
dev_size = int(0.1 * len(data))
test_size = int(0.1 * len(data))
valid = data[:dev_size]
test = data[dev_size: dev_size + test_size]
train = data[dev_size + test_size:]

print('train', len(train))
with open('./train_data/esc-train.txt', 'w') as f:
    for e in train:
        f.write(json.dumps(e) + '\n')
with open('./train_data/esc-sample.json', 'w') as f:
    json.dump(train[:10], f, ensure_ascii=False, indent=2)

print('valid', len(valid))
with open('./train_data/esc-valid.txt', 'w') as f:
    for e in valid:
        f.write(json.dumps(e) + '\n')

print('test', len(test))
with open('./train_data/esc-test.txt', 'w') as f:
    for e in test:
        f.write(json.dumps(e) + '\n')
