import torch

from env import Env
from agent import MDDP,PG
from utils import *
from itertools import count
from tqdm import tqdm
import argparse
from transformers import BertTokenizer, RobertaTokenizer, BertConfig, RobertaConfig,AutoTokenizer,AutoConfig,AutoModelForCausalLM
from fastchat.model import add_model_args
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
from data.prompt import behaviors,stages
from knowledge.preprocess_add_knowledge import get_strategy_category,search_faiss_and_fetch_text
EOS_TOKEN_ID=2
BOS_TOKEN_ID=1

tok = {'bert': BertTokenizer, 'roberta': RobertaTokenizer}
cfg = {'bert': BertConfig, 'roberta': RobertaConfig}
# ESConvAct = {"Question": ["context-related","Please ask the Seeker to elaborate on the situation they just described based on the context."],
#             "Self-disclosure": ["information-related","Please provide a statement relating to the Seeker about the context they just described with the help of provided knowledge."],
#             "Affirmation and Reassurance": ["emotion-related","Please provide affirmation and reassurance to the Seeker on the context they just described with the help of provided knowledge."],
#             "Providing Suggestions": ["information-related","Please provide suggestion to the Seeker on the context they just described with the help of provided knowledge."],
#             "Others": ["context-related","Please chat with the seeker."],
#             "Reflection of feelings": ["emotion-related","Please acknowledge the Seeker's feelings about the context they described with the help of provided knowledge."],
#             "Information": ["information-related","Please provide factual information to help the Seeker with their situation with the help of provided knowledge ."],
#              "Restatement or Paraphrasing": ["context-related","Please acknowledge the Seeker's feelings by paraphrasing their situation."]
#              }
ESConvAct = {"Question": ["context-related","You are an emotional support agent as Supporter in the conversation, please ask the Seeker to elaborate on the situation they described with the help of provided knowledge."],
            "Self-disclosure": ["information-related","You are an emotional support agent as Supporter in the conversation, please provide a statement relating to the Seeker about the situation they just described."],
            "Affirmation and Reassurance":  ["emotion-related","You are an emotional support agent as Supporter in the conversation, please provide affirmation and reassurance to the Seeker on the situation they described with the help of provided knowledge."],
            "Providing Suggestions": ["information-related","You are an emotional support agent as Supporter in the conversation, please provide suggestion to the Seeker on the situation they described with the help of provided knowledge."],
            "Others": ["context-related","You are an emotional support agent as Supporter in the conversation, please chat with the Seeker with the help of provided knowledge."],
            "Reflection of feelings":  ["emotion-related","You are an emotional support agent as Supporter in the conversation, please acknowledge the Seeker's feelings about the situation they described with the help of provided knowledge."],
            "Information":["information-related", "You are an emotional support agent as Supporter in the conversation, please provide factual information to help the Seeker with their situation with the help of provided knowledge."],
            "Restatement or Paraphrasing": ["context-related","You are an emotional support agent as Supporter in the conversation, please acknowledge the Seeker's feelings by paraphrasing their situation with the help of provided knowledge."]}

import string
def clean_knowledge(text):
    # 保留英文字母（a-zA-Z）、标点符号和空格
    allowed = string.ascii_letters + string.punctuation + ' '
    return ''.join(ch for ch in text if ch in allowed)

def train(args, config, dataset, filename, tokenizer,embed_model):
    env = Env(args, dataset, mode='train') # env init
    set_random_seed(args.seed)
    dialog_planner = MDDP(args, config, tokenizer) # policy network init
    if args.PG==True:
        if 'vicuna' in [args.system] or 'llama2' in [args.system]:
            generation_config=AutoConfig.from_pretrained(args.model_path)
            prompt_generator = PG(args, generation_config, env.vicuna_tokenizer)
    else:
        print("No bridge model for prompt generation")
    # load policy parameters
    if args.dp_sft_dir is not None:
        print('Staring loading policy model from {}'.format(args.dp_sft_dir))
        dialog_planner.load_model(data_name=args.data_name, filename=args.dp_sft_dir)
        dialog_planner.to(args.device)
        print(next(dialog_planner.parameters()).device, flush=True)

    if args.load_rl_epoch > 0:
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        dialog_planner.load_model(data_name=args.data_name, filename=args.dp_sft_dir, epoch_user=args.load_rl_epoch)
        dialog_planner.to(args.device)


    if args.pg_sft_dir is not None and args.PG == True:
        print('Staring loading pg model from {}'.format(args.pg_sft_dir))
        prompt_generator.load_model(data_name=args.data_name, filename=args.pg_sft_dir)
        prompt_generator.to(args.device)
        if args.if_float16 == True:
            prompt_generator.to(torch.float16)
        print(next(dialog_planner.parameters()).device, flush=True)
    else:
        prompt_generator = None
    if args.load_rl_epoch > 0:
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        prompt_generator.load_model(data_name=args.data_name, filename=args.pg_sft_dir, epoch_user=args.load_rl_epoch)
        prompt_generator.to(args.device)

    test_performance = []
    if args.do_eval:
        SR15_mean = evaluate(args, dataset, dialog_planner,prompt_generator, filename, 0, env,embed_model)
        test_performance = [SR15_mean]
    if not args.do_train:
        return
    for train_step in range(1, args.max_steps+1):
        SR, AvgT, total_reward = 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        response_loss = torch.tensor(0, dtype=torch.float, device=args.device)
        for i_episode in tqdm(range(args.sample_times),desc='sampling'):
            #blockPrint()
            print('\n================new tuple:{}===================='.format(i_episode))
            dialog_planner.reset()
            prompt_generator.reset()
            conversation,states = env.reset()
            epi_reward = 0
            done = False
            for t in count():  # user  dialog
                action= dialog_planner.select_action(conversation, int(states[-1]['emotion']) - 1,
                                                                int(states[-1]['trust']) - 1,
                                                           stages.index(states[-1]['stage']),
                                                                behaviors.index(states[-1]['behavior']),
                                                                is_test=True)
                if args.RAG == True:
                    category = get_strategy_category(action)
                    if category == "information-related":
                        k = search_faiss_and_fetch_text(embed_model, conversation[-4:], args.psyqa_index_path,
                                                        args.psyqa_db_path)
                        retrieved_knowledge = category + " knowledge: " + k[0][1]
                    elif category == "emotion-related":
                        k = search_faiss_and_fetch_text(embed_model, conversation[-4:], args.emotion_index_path,
                                                        args.emotion_db_path)
                        retrieved_knowledge = category + " knowledge: " + "context: " + k[0][0] + " response: " + k[0][1]
                    # retrieved_knowledge = ' '.join(k)
                    elif category == "context-related":
                        k = generate_context_knowledge(conversation, env.vicuna_model, args.max_new_tokens,
                                                       args.temperature, args.device)
                        retrieved_knowledge = category + " knowledge: " + k
                    else:
                        "Something wrong with category"
                else:
                    retrieved_knowledge = None
                if args.PG == True and args.RAG == True:
                    retrieved_knowledge = clean_knowledge(retrieved_knowledge)
                    dial = ' '.join([f"{turn['role']}: {turn['content']}" for turn in conversation])
                    gen_input_ids = env.vicuna_tokenizer.encode(dial)
                    gen_input_ids = gen_input_ids[-args.max_seq_length + 1:]+ [EOS_TOKEN_ID]
                    gen_input_ids = torch.tensor([gen_input_ids]).long().to(args.device)
                    knowledge_ids = env.vicuna_tokenizer.encode(retrieved_knowledge)[
                                    :args.max_seq_length]+[EOS_TOKEN_ID]
                    knowledge_ids = torch.tensor([knowledge_ids]).long().to(args.device)
                    instruction = ESConvAct[action]
                    instruction_ids = env.vicuna_tokenizer.encode(instruction)+[EOS_TOKEN_ID]
                    instruction_ids = torch.tensor([instruction_ids]).long().to(args.device)
                    with torch.no_grad():
                        context_embeddings = env.vicuna_model.model.embed_tokens(
                            gen_input_ids)  # [1, seq_len, hidden_size]
                        knowledge_embeddings = env.vicuna_model.model.embed_tokens(knowledge_ids)
                        instruction_embeddings = env.vicuna_model.model.embed_tokens(instruction_ids)
                        First_embedding = env.vicuna_model.model.embed_tokens(
                            torch.tensor([[BOS_TOKEN_ID]]).long().to(args.device))
                    if args.pg_rl==True:
                        embedding,_,_ = prompt_generator(
                            context_embeddings=context_embeddings,
                            attention_mask=None,
                            knowledge_embeddings=knowledge_embeddings,
                            instruction_embeddings=instruction_embeddings,
                            train_mode='rl'
                        )
                    else:
                        with torch.no_grad():
                            embedding = prompt_generator(
                                context_embeddings=context_embeddings,
                                attention_mask=None,
                                knowledge_embeddings=knowledge_embeddings,
                                instruction_embeddings=instruction_embeddings,
                                train_mode='test'
                            )

                    prompt_embedding = torch.cat([embedding, First_embedding.view(1, 1, -1)], dim=1)
                else:
                    prompt_embedding = None
                    # pg_response = None
                states, conversation, reward, response_reward, done = env.step(action, retrieved_knowledge,
                                                                                    prompt_embedding)  # state tracking and self-play
                # reward=policy.compute_reward(int(states[-1]['emotion']),states[-1]['trust'])
                epi_reward += reward+response_reward
                if args.pg_rl==True:
                    response_reward=response_reward#+reward
                    response_reward = torch.tensor([response_reward], device=args.device, dtype=torch.float)
                    prompt_generator.rewards.append(response_reward)
                if done:
                    if done == 1:
                        SR += 1
                    AvgT += t+1
                    total_reward += epi_reward
                    break
            if args.pg_rl==True:
                print("Optimize Prompt Generator")
                new_response_loss=prompt_generator.optimize_PPO()
                print(new_response_loss,True)
                response_loss += new_response_loss
        enablePrint() # Enable print function
        print('loss : {} in epoch_uesr {}'.format(loss.item()/args.sample_times, args.sample_times))
        print('response_loss : {} in epoch_uesr {}'.format(response_loss.item() / args.sample_times, args.sample_times))
        print('SR:{}, AvgT:{}, rewards:{} Total epoch_uesr:{}'.format(SR / args.sample_times,
                    AvgT / args.sample_times, total_reward / args.sample_times, args.sample_times))
        if train_step % args.save_num == 0:
            if args.pg_rl == True:
                prompt_generator.save_model(data_name=args.data_name, filename=filename+"pg", epoch_user=train_step)
        if train_step % args.eval_num == 0:
            SR_all = evaluate(args, dataset, dialog_planner, prompt_generator, filename, train_step, env, embed_model)
            test_performance.append(SR_all)


def evaluate(args, dataset, dialogue_planner,prompt_generator, filename, i_episode, train_env,embed_model):
    if 'vicuna' in [args.system, args.user, args.critic] or 'llama2' in [args.system, args.user, args.critic]:
        test_env = Env(args, dataset, mode='test', env_model=train_env.vicuna_model, env_tokenizer=train_env.vicuna_tokenizer)
    else:
        test_env = Env(args, dataset, mode='test') # env init
    test_env.vicuna_model=test_env.vicuna_model
    set_random_seed(args.seed)

    SR, AvgT, total_reward = 0, 0, 0
    SR_turn = [0]* args.max_turn
    test_size = len(test_env.dataset)
    print('Test size: ', test_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    record_filename = 'Record-epoch-{}-'.format(i_episode) + filename
    REC_PATH = TMP_DIR[args.data_name] + '/eval_result/' + record_filename + '.txt'
    if not os.path.isdir(TMP_DIR[args.data_name] + '/eval_result/'):
        os.makedirs(TMP_DIR[args.data_name] + '/eval_result/')
    rec_file = open(REC_PATH, 'w')
    for test_num in tqdm(range(test_size)):  #test_size
        #blockPrint()
        print('\n================test tuple:{}===================='.format(test_num))
        epi_reward = 0
        done = 0
        is_last_turn = False
        conversation,states = test_env.reset()
        for t in count():  # user  dialog
            if args.state_attention == True:
                emotion=int(states[-1]['emotion']) - 1
                trust=int(states[-1]['trust']) - 1
                stage=stages.index(states[-1]['stage'])
                behavior=behaviors.index(states[-1]['behavior'])
            else:
                emotion=None
                trust=None
                stage=None
                behavior=None
            action= dialogue_planner.select_action(conversation, emotion, trust, stage, behavior,
                                              is_test=True)#use MDDP to predict action
            if args.RAG == True:
                category = get_strategy_category(action)
                if category == "information-related":
                    k = search_faiss_and_fetch_text(embed_model, conversation[-4:], args.psyqa_index_path,
                                                    args.psyqa_db_path)
                    retrieved_knowledge = category + " knowledge: " + k[0][1]
                elif category == "emotion-related":
                    k = search_faiss_and_fetch_text(embed_model, conversation[-4:], args.emotion_index_path,
                                                    args.emotion_db_path)
                    retrieved_knowledge = category + " knowledge: " + "context: " + k[0][0] + " reponse: " + k[0][1]
                elif category == "context-related":
                    k = generate_context_knowledge(conversation, test_env.vicuna_model, args.max_new_tokens,
                                                   args.temperature, args.device)
                    retrieved_knowledge = category + " knowledge: " + k
                else:
                    "Something wrong with category"
                retrieved_knowledge = clean_knowledge(retrieved_knowledge)
                if args.PG==True:
                    knowledge_ids = test_env.vicuna_tokenizer.encode(retrieved_knowledge)[:args.max_seq_length] + [
                        EOS_TOKEN_ID]
                    knowledge_ids = torch.tensor([knowledge_ids]).long().to(args.device)
                    with torch.no_grad():
                        knowledge_embeddings = test_env.vicuna_model.model.embed_tokens(knowledge_ids)
            else:
                retrieved_knowledge=None
                knowledge_embeddings=None
            if args.PG == True:
                dial = ' '.join([f"{turn['role']}: {turn['content']}" for turn in conversation])
                gen_input_ids = test_env.vicuna_tokenizer.encode(dial)
                gen_input_ids = gen_input_ids[-args.max_seq_length+1:]+ [EOS_TOKEN_ID]
                gen_input_ids = torch.tensor([gen_input_ids]).long().to(args.device)
                instruction = ESConvAct[action]
                instruction_ids = test_env.vicuna_tokenizer.encode(instruction)+[EOS_TOKEN_ID]
                instruction_ids = torch.tensor([instruction_ids]).long().to(args.device)
                with torch.no_grad():
                    context_embeddings = test_env.vicuna_model.model.embed_tokens(gen_input_ids)  # [1, seq_len, hidden_size]
                    instruction_embeddings = test_env.vicuna_model.model.embed_tokens(instruction_ids)
                    First_embedding = test_env.vicuna_model.model.embed_tokens(
                        torch.tensor([[BOS_TOKEN_ID]]).long().to(args.device))
                embedding = prompt_generator(
                    context_embeddings=context_embeddings,
                    attention_mask=None,
                    knowledge_embeddings=knowledge_embeddings,
                    instruction_embeddings=instruction_embeddings,
                    train_mode="test"
                )

                prompt_embedding = torch.cat([embedding, First_embedding.view(1, 1, -1)], dim=1)
            else:
                prompt_embedding=None
                # pg_response = None
            states, conversation,reward,response_reward, done = test_env.step(action,retrieved_knowledge,prompt_embedding)#state tracking and self-play
            # reward=policy.compute_reward(int(states[-1]['emotion']),states[-1]['trust'])
            if args.data_name == 'cb' and reward < 0: # reward = Sale-to-List Ratio
                reward = 0
            epi_reward += reward+response_reward
            if done:
                if done == 1:  
                    SR_turn = [v+1 if i>t  else v for i, v in enumerate(SR_turn) ]
                    SR += 1
                total_reward += epi_reward
                AvgT += t+1

                rec_file.write('%s\n\n' % str({'dialog':states[-1], 'reward':epi_reward}))
                break
        enablePrint()
    
    SR_mean = float(SR)/test_size
    AvgT_mean = float(AvgT)/test_size
    reward_mean = total_reward/test_size
    SR_all = [SR_mean, AvgT_mean, reward_mean]
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=test_num, SR=SR_all, mode='test')  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = float(SR_turn[i])/test_size
    print('success turn:{}'.format(SRturn_all))
    print('SR:{}, AvgT:{}, reward:{}'.format(SR_mean, AvgT_mean, reward_mean))
    PATH = TMP_DIR[args.data_name] + '/eval_result/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(test_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')
    with open(PATH, 'a') as f:
        f.write('{}\t{}\t{}\t{}\n'.format(i_episode, SR_mean, AvgT_mean, reward_mean))
    return SR_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--num_gpus', type=int, default=2, help='number of gpus.')
    # parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--dp_learning_rate', type=float, default=1e-6, help='learning rate.')
    parser.add_argument('--pg_learning_rate', type=float, default=1e-6, help='learning rate.')
    parser.add_argument('--data_name', type=str, default='esc', choices=['esc','cima','cb'],
                        help='One of {esc, cima, cb}.')
    parser.add_argument('--system', type=str, default='llama2', choices=['vicuna','chatgpt','llama2','chat_ai'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--user', type=str, default='chat_ai', choices=['vicuna','chatgpt','llama2','chat_ai'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--critic', type=str, default='chat_ai', choices=['vicuna','chatgpt','llama2','chat_ai'],
                        help='One of {vicuna, chatgpt, llama2}.')
    parser.add_argument('--dp_sft_dir', default='', #../pretrain/outputs/best_pretrain.pt
                        type=str, help="Pretrain model path.")#sft/esc/roberta/Truestate_attention/Truerag/best_checkpoint/dialogue_planner#
    parser.add_argument('--pg_sft_dir', default='',
                                         type=str, help="Pretrain model path.")
    parser.add_argument("--reframing_index_path", type=str, default="./knowledge/reframing_knowledge_faiss.index",
                        help="Path to FAISS index for Reframing")
    parser.add_argument("--reframing_db_path", type=str, default="./knowledge/reframing_knowledge.db",
                        help="CSV file for Reframing knowledge")

    parser.add_argument("--psyqa_index_path", type=str, default="./knowledge/psyqa_knowledge_faiss.index",
                        help="Path to FAISS index for PsyQA")
    parser.add_argument("--psyqa_db_path", type=str, default="./knowledge/psyqa_knowledge.db", help="CSV file for PsyQA knowledge")

    parser.add_argument("--emotion_index_path", type=str, default="./knowledge/emotional_knowledge_faiss.index",
                        help="Path to FAISS index for Emotion-Reflection")
    parser.add_argument("--emotion_db_path", type=str, default="./knowledge/emotional_knowledge.db",
                        help="CSV file for Emotion-Reflection")

    parser.add_argument('--max_turn', type=int, default=10, help='max conversation turn')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--load_rl_epoch', type=int, default=0, help='load agent from epoch')

    parser.add_argument("--cache_dir", default='./storage_fast/plm', type=str, help="The cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--debug", default=False,action="store_true")
    parser.add_argument("--model_path", type=str, default="./../llama2-7b-chat-hf")
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--model_name_or_path", default='./../roberta-large', type=str, help="model name or path")
    parser.add_argument("--chat_ai_model", type=str, default="llama-3.3-70b-instruct")#qwen2.5-72b-instruct #meta-llama-3.1-70b-instruct #deepseek-r1-distill-llama-70b
    parser.add_argument("--do_lower_case",default=True, help="Set this flag if you are using an uncased model.")

    parser.add_argument('--max_steps', type=int, default=5, help='max training steps')
    parser.add_argument('--sample_times', type=int, default=80, help='the epoch of sampling')
    parser.add_argument('--eval_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--save_num', type=int, default=1, help='the number of steps to save RL model and metric')
    # RAG
    parser.add_argument("--embed_model_path", type=str, default='')
    parser.add_argument("--RAG", type=bool, default=True)
    parser.add_argument("--PG", type=bool, default=True)
    parser.add_argument("--state_attention", default=False, type=bool,
                        help="if add four state aspects")
    parser.add_argument("--do_train", default=False, help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, help="Whether to run eval.")
    parser.add_argument('--eps_clip', type=float, default=0.2, help='clip')
    parser.add_argument("--pg_rl", type=bool, default=True, help="Whether to run RL training for PG model.")
    # RL stage: DPO--PPO
    # generation
    parser.add_argument('--temperature', type=float, default=0.7)
    # parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--eos_token_id', type=int, default=2)
    parser.add_argument('--if_float16', type=bool, default=False)

    add_model_args(parser)
    args = parser.parse_args()
    print(args)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    dataset = load_dataset(args.data_name)
    filename = '{}-{}-{}-{}'.format(args.data_name,args.system,args.user,args.critic)
    print(filename,flush=True)
    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir)

    if args.RAG == True:
        embed_model = SentenceTransformer(args.embed_model_path)
    else:
        embed_model=None
    train(args, config, dataset, filename, tokenizer,embed_model)

if __name__ == '__main__':
    main()