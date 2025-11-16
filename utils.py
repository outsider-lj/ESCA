import pickle
import numpy as np
import random
import torch
import os
import sys
import time
import openai
TMP_DIR = {
    'esc': './tmp/esc',
    'cima': './tmp/cima',
    'cb': './tmp/cb',
}

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter

def compute_bleu4(preds, refs):
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    for pred, ref in zip(preds, refs):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        bleu_scores.append(bleu)
    return sum(bleu_scores) / len(bleu_scores)


def compute_distinct2(sentences):
    bigram_counter = Counter()
    total_bigrams = 0

    for sentence in sentences:
        tokens = sentence.strip().split()
        bigrams = list(zip(tokens, tokens[1:]))
        bigram_counter.update(bigrams)
        total_bigrams += len(bigrams)

    if total_bigrams == 0:
        return 0.0
    return len(bigram_counter) / total_bigrams

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def load_dataset(data_name):
    dataset = {'train':[], 'test':[], 'valid':[]}
    for key in dataset:
        with open("./data/train_data/%s-%s.txt"%(data_name, key),'r') as infile:
            for line in infile:
                dataset[key].append(eval(line.strip('\n')))
    return dataset


def set_cuda(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    devices_id = [int(device_id) for device_id in args.gpu.split()]
    device = (
        torch.device("cuda:{}".format(str(devices_id[0])))
        if use_cuda
        else torch.device("cpu")
    )
    return device, devices_id
def query_chat_ai_model(api_key: str, messages: str, url: str, model: str , max_tokens: int = 128, temperature: float = 0, n: int = 1):
    client = openai.OpenAI(
        api_key=api_key,
        base_url=url
    )
    flag = True

    while flag:
        try:
            # print("request")
            completions = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                #request_timeout=10,
            )
            # print("completions:",completions)
            if n == 1:
                output = completions.choices[0].message.content.strip()
            else:
                output = []
                for choice in completions.choices:
                    output.append(choice.message.content.strip())

            flag = False
        except Exception as e:
            print("Some error happened here.")
            time.sleep(5)
    return output
def generate_context_knowledge(history,model,tokenizer,max_new_tokens,temperature,device):
    prompt = "Based on the given dialogue history, what does the current dialogue focus on? And what important information related to this topic not be metioned" \
             "The answer is less than 100 words and without any format and analysis."
    try:
        user_input = "\n".join([f"{t['role']}: {t['content']}" for t in history])
    except:
        user_input=" ".join(history)
    messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}]
    # basellm
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([prompt]).input_ids
    input_ids = torch.tensor(input_ids).long().to(device)
    output_ids =model.generate(
        input_ids,
        # max_new_tokens=200,
        temperature=temperature,
        do_sample=True,
        early_stopping=True,
        # num_return_sequences=10,
    )
    # outputs = []
    # for o in output_ids:
    output_ids = output_ids[0][len(input_ids[0]):]
    knowledge = tokenizer.decode(output_ids, skip_special_tokens=True,
                                          spaces_between_special_tokens=False)
    return knowledge

def save_rl_mtric(dataset, filename, epoch, SR, mode='train'):
    PATH = TMP_DIR[dataset] + '/eval_result/' + filename + '.txt'
    if not os.path.isdir(TMP_DIR[dataset] + '/eval_result/'):
        os.makedirs(TMP_DIR[dataset] + '/eval_result/')
    if mode == 'train':
        with open(PATH, 'a') as f:
            f.write('===========Train===============\n')
            f.write('Starting {} user epochs\n'.format(epoch))
            f.write('training SR: {}\n'.format(SR[0]))
            f.write('training Avg@T: {}\n'.format(SR[1]))
            f.write('training Rewards: {}\n'.format(SR[2]))
            f.write('================================\n')
            # f.write('1000 loss: {}\n'.format(loss_1000))
    elif mode == 'test':
        with open(PATH, 'a') as f:
            f.write('===========Test===============\n')
            f.write('Testing {} user tuples\n'.format(epoch))
            f.write('Testing SR: {}\n'.format(SR[0]))
            f.write('Testing Avg@T: {}\n'.format(SR[1]))
            f.write('Testing Rewards: {}\n'.format(SR[2]))
            f.write('================================\n')