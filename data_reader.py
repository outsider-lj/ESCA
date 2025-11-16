import os
import logging

import numpy as np
import torch
import pickle
# from prompt import ESConvAct
import json
from data.prompt import stages,behaviors
logger = logging.getLogger(__name__)
ESConvAct = {"Question": "You are a Supporter in the emotional support conversation, please ask the Seeker to elaborate on the situation they described with the help of provided knowledge.",
            "Self-disclosure": "You are a Supporter in the emotional support conversation, please provide a statement relating to the Seeker about the situation they just described.",
            "Affirmation and Reassurance": "You are a Supporter in the emotional support conversation, please provide affirmation and reassurance to the Seeker on the situation they described with the help of provided knowledge.",
            "Providing Suggestions": "You are a Supporter in the emotional support conversation, please provide suggestion to the Seeker on the situation they described with the help of provided knowledge.",
            "Others": "You are a Supporter in the emotional support conversation, please chat with the Seeker with the help of provided knowledge.",
            "Reflection of feelings": "You are a Supporter in the emotional support conversation, please acknowledge the Seeker's feelings about the situation they described with the help of provided knowledge.",
            "Information": "You are a Supporter in the emotional support conversation, please provide factual information to help the Seeker with their situation with the help of provided knowledge.",
            "Restatement or Paraphrasing": "You are a Supporter in the emotional support conversation, please acknowledge the Seeker's feelings by paraphrasing their situation with the help of provided knowledge."}
Knowledge_Mapping = {
    "information-related": ["self-disclosure", "information", "providing suggestions"],
    "emotion-related": ["affirmation and reassurance", "reflection of feelings"],
    "context-related": ["restatement or paraphrasing", "question", "others"]
}
def get_strategy_category(strategy):
    for cat, strategies in Knowledge_Mapping.items():
        if strategy.lower() in strategies:
            return cat
    return None
role_map = {'esc': {'supporter': 'Supporter', 'seeker': 'Seeker'},}
act_map = {'esc': ESConvAct}
EOS_TOKEN_ID=2
BOS_TOKEN_ID=1
ROBERT_BOS_TOKEN_ID=0

def conduct_instruction(strategy):
    prompt=ESConvAct[strategy]
    return prompt

def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def clean_text(text):
    text=text.lower()
    text=text.strip('\n')
    text=text.strip()
    return text
import string
def clean_knowledge(text):

    allowed = string.ascii_letters + string.punctuation + ' '
    return ''.join(ch for ch in text if ch in allowed)

def load_and_cache_examples(args, tokenizer,generation_tokenizer, evaluate=False):
    mode = args.set_name if evaluate else 'train'
    print(mode)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'sft_{}_{}_{}_{}_{}'.format(
        args.data_name,
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),str(args.generation_model_name)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = read_pkl(cached_features_file)
        print("Loaded number of instance:", len(features['source_ids']))
    else:

        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = convert_to_features(args, tokenizer,generation_tokenizer, mode)
        print("Loaded number of instance:", len(features['source_ids']))
    
        logger.info("Saving features into cached file %s", cached_features_file)
        write_pkl(features, cached_features_file)
    return features

def convert_to_features(args, tokenizer,generation_tokenizer, mode): #将对话转化为
    path = os.path.join(args.data_dir,'train_data','{}-{}.txt'.format(args.data_name, mode))
    act = sorted(list(act_map[args.data_name].keys()))
    print('tokenizing {}'.format(path))
    with open(path, 'r', encoding='utf-8') as infile:
        max_dia_len = 0
        avg_dia_len = []
        source_ids = []
        strategy_ids = []
        emotion_intensity=[]
        trust_scores=[]
        stage_ids=[]
        behavior_ids=[]
        gen_source_ids = []
        response_ids=[]
        if mode !="test":
            knowledge_ids = []
            instruction_ids=[]
        # relaibility_scores=[]
        if args.data_name in ['esc','cb']:
            for line in infile:
                sample = json.loads(line.strip('\n'))
                dial = sample['dialog']
                state = []#每一轮的话语
                generation_state=[]
                # supporter_ids = generation_tokenizer.encode("Supporter: ")[1:-1]
                for i,turn in enumerate(dial):
                    if turn['speaker'] == 'supporter' and len(state) > 0:
                        dial_id = []
                        gen_dial_id=[]
                        for s in state[::-1]:
                            if len(dial_id) + len(s) > args.max_seq_length:
                                break
                            dial_id = s[1:] + dial_id
                        for gs in generation_state[::-1]:
                            if len(gen_dial_id) + len(gs) > args.max_seq_length:
                                break
                            gen_dial_id = gs[1:] + gen_dial_id
                        source_id = s[:1] + dial_id+[EOS_TOKEN_ID]# 对话历史拼接
                        gen_source_id=gs[:1]+gen_dial_id+[EOS_TOKEN_ID]#generation_tokenizer("Supporter: ")[1:-1]+
                        max_dia_len = max(max_dia_len, len(source_id))
                        a = 1
                        while dial[i - a]['speaker'] != 'seeker':
                            a = a + 1
                        # try:
                        stage=dial[i - a]["stage_of_change"].split(',')
                        behavior = dial[i - a]["behavior"].split(',')
                        if behavior[0].lower()=="statement-agreement":
                            behavior[0]="agreement"
                        if behavior[0].lower()=="other":
                            behavior[0]="others"
                        stage_id=stages.index(stage[0].lower())
                        behavior_id=behaviors.index(behavior[0].lower())
                        e = int(dial[i - a]['emotion_intensity']) - 1
                        trust_score=int(dial[i - a]['trust']) - 1
                        strategy_id = act.index(turn['strategy'])  # 策略序号
                        response_id = generation_tokenizer.encode(turn["text"])#+[EOS_TOKEN_ID]
                        if mode != "test":
                            retreived_knowledge_lst=dial[i]["retrieved_knowledge"]
                            category=get_strategy_category(turn['strategy'])
                            if category=="context-related":
                                retreived_knowledge = category + " knowledge: " + clean_knowledge(
                                    retreived_knowledge_lst[0])
                            elif category=="emotion-related":
                                retreived_knowledge = category + " knowledge: " +'context: '+ clean_knowledge(
                                    retreived_knowledge_lst[0])+'response: '+ clean_knowledge(
                                    retreived_knowledge_lst[1])
                            elif category == "information-related":
                                retreived_knowledge = category + " knowledge: "+clean_knowledge(retreived_knowledge_lst[1])
                            else:
                                print("Something error in knowledge retrieving")
                            knowledge_input_id = generation_tokenizer.encode(retreived_knowledge)[:args.max_seq_length]+[EOS_TOKEN_ID]
                            instruction=conduct_instruction(turn['strategy'])
                            instruction_id = generation_tokenizer.encode(instruction)+[EOS_TOKEN_ID]
                        stage_ids.append(stage_id)
                        behavior_ids.append(behavior_id)
                        emotion_intensity.append(e)
                        trust_scores.append(trust_score)
                        source_ids.append(source_id[-args.max_seq_length + 1:])
                        strategy_ids.append(strategy_id)
                        avg_dia_len.append(len(source_id))
                        response_ids.append(response_id)
                        gen_source_ids.append(gen_source_id)
                        if mode != "test":
                            knowledge_ids.append(knowledge_input_id)
                            instruction_ids.append(instruction_id)
                    state.append(tokenizer.encode("%s: %s" % (role_map[args.data_name][turn['speaker']], turn['text'])))
                    generation_state.append(generation_tokenizer.encode("%s: %s" % (role_map[args.data_name][turn['speaker']], turn['text'])))
                   # input_ids=generation_tokenizer.encode("%s: %s" % (role_map[args.data_name][turn['speaker']], turn['text']))

        print('{} set, max_dia_len: {}, avg_dia_len: {}'.format(mode, max_dia_len, float(sum(avg_dia_len))/len(avg_dia_len)))
    if mode=="test":
        return {'source_ids': source_ids, 'strategy_ids': strategy_ids, 'emotion': emotion_intensity,
                    'trust': trust_scores,
                    "stage": stage_ids, "behavior": behavior_ids,
                    "response_ids": response_ids, "gen_source_ids": gen_source_ids}
    else:
        return {'source_ids':source_ids, 'strategy_ids':strategy_ids,'emotion':emotion_intensity,'trust':trust_scores,
            "stage":stage_ids,"behavior":behavior_ids,"retrieved_knowledge":knowledge_ids,"response_ids":response_ids,"gen_source_ids":gen_source_ids,"instruction_ids":instruction_ids}

def convert_to_features_for_dp(args, tokenizer,generation_tokenizer, mode): #将对话转化为
    path = os.path.join(args.data_dir,'train_data','{}-{}.txt'.format(args.data_name, mode))
    act = sorted(list(act_map[args.data_name].keys()))
    print('tokenizing {}'.format(path))
    with open(path, 'r', encoding='utf-8') as infile:
        max_dia_len = 0
        avg_dia_len = []
        source_ids = []
        strategy_ids = []
        emotion_intensity=[]
        trust_scores=[]
        stage_ids=[]
        behavior_ids=[]
        # relaibility_scores=[]
        if args.data_name in ['esc','cb']:
            for line in infile:
                sample = json.loads(line.strip('\n'))
                dial = sample['dialog']
                state = []#每一轮的话语
                generation_state=[]
                supporter_ids = generation_tokenizer.encode("Supporter: ")[1:-1]
                print(supporter_ids,flush=True)
                for i,turn in enumerate(dial):
                    if turn['speaker'] == 'supporter' and len(state) > 0:
                        dial_id = []
                        for s in state[::-1]:
                            if len(dial_id) + len(s) > args.max_seq_length:
                                break
                            dial_id = s[1:] + dial_id
                        source_id = s[:1] + dial_id+[EOS_TOKEN_ID]# 对话历史拼接
                        max_dia_len = max(max_dia_len, len(source_id))
                        a = 1
                        while dial[i - a]['speaker'] != 'seeker':
                            a = a + 1
                        try:
                            stage=dial[i - a]["stage_of_change"].split(',')
                            behavior = dial[i - a]["behavior"].split(',')
                            if behavior[0].lower()=="statement-agreement":
                                behavior[0]="agreement"
                            stage_id=stages.index(stage[0].lower())
                            behavior_id=behaviors.index(behavior[0].lower())
                            e = int(dial[i - a]['emotion_intensity']) - 1
                            trust_score=int(dial[i - a]['trust']) - 1
                            strategy_id = act.index(turn['strategy'])  # 策略序号
                            stage_ids.append(stage_id)
                            behavior_ids.append(behavior_id)
                            emotion_intensity.append(e)
                            trust_scores.append(trust_score)
                            source_ids.append(source_id[-args.max_seq_length + 1:])
                            strategy_ids.append(strategy_id)
                            avg_dia_len.append(len(source_id))

                            # find related seeker's state

                        except:
                            continue
                        # relaibility_scores.append(dial[i-1]["relaibility"])

                    state.append(tokenizer.encode("%s: %s" % (role_map[args.data_name][turn['speaker']], clean_text(turn['text']))))
                    generation_state.append(generation_tokenizer.encode("%s: %s" % (role_map[args.data_name][turn['speaker']], clean_text(turn['text']))))
                   # input_ids=generation_tokenizer.encode("%s: %s" % (role_map[args.data_name][turn['speaker']], turn['text']))
                    #encoder时包含了<s>
        elif args.data_name == 'cima':
            for line in infile:
                sample = eval(line.strip('\n'))
                dial = sample['dialog']
                state = []

                target_id = act.index(sample['strategy'])
                dial_id = []
                for s in dial:
                    s = tokenizer.encode("%s: %s" % (role_map[args.data_name][s['speaker']], s['text']))
                    dial_id += s[1:]
                source_id = s[:1] + dial_id
                source_ids.append(source_id[-args.max_seq_length+1:])
                strategy_ids.append(target_id)

                avg_dia_len.append(len(source_id))
                max_dia_len = max(max_dia_len, len(source_id))


        print('{} set, max_dia_len: {}, avg_dia_len: {}'.format(mode, max_dia_len, float(sum(avg_dia_len))/len(avg_dia_len)))

    return {'source_ids':source_ids, 'strategy_ids':strategy_ids,'emotion':emotion_intensity,'trust':trust_scores,
            "stage":stage_ids,"behavior":behavior_ids}
