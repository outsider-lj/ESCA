import argparse

import numpy as np
from torch.optim import AdamW
from transformers import AutoConfig, LlamaConfig,LlamaTokenizer, RobertaTokenizer, BertConfig, RobertaConfig,AutoModelForCausalLM,AutoTokenizer
from sentence_transformers import SentenceTransformer
import glob
import logging
import os
import random
from pytorch_transformers import WarmupLinearSchedule
import torch
import utils
import data_reader
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from utils import compute_bleu4,compute_distinct2,load_dataset
from agent import MDDP,PG
from prompt import ESConvAct
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import generate_context_knowledge
from knowledge.preprocess_add_knowledge import search_faiss_and_fetch_text,get_strategy_category
import re
# from metric import NLGEval
# python sft_dp.py --gpu="0 1" --do_train --overwrite_output_dir --per_gpu_train_batch_size=8 --per_gpu_eval_batch_size=8
EOS_TOKEN_ID=2
BOS_TOKEN_ID=1
tok = {'llama2': LlamaTokenizer,  'roberta': RobertaTokenizer}
cfg = {'llama2': LlamaConfig, 'roberta': RobertaConfig}
ACT = sorted(list(ESConvAct.keys()))
from prompt import vicuna_prompt

def generate_response( args,model,tokenizer, messages):
    prompt = vicuna_prompt(messages,'system')
    input_ids = tokenizer([prompt]).input_ids
        # print(len(input_ids[0]))
    max_new_tokens =args.max_new_tokens
    output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            early_stopping=True
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    # output =tokenizer.decode(output_ids, skip_special_tokens=True,
    #                                           spaces_between_special_tokens=False)
    return output_ids
def generate_from_embedding(args, model, inputs_embeds: torch.Tensor):
    # device = inputs_embeds.device
    allowed_keys = {
        'max_length', 'min_length', 'temperature', 'top_k', 'top_p', 'num_beams',
        'do_sample', 'repetition_penalty', 'length_penalty', 'bad_words_ids',
        'no_repeat_ngram_size', 'early_stopping',
        'use_cache', 'bos_token_id', 'eos_token_id'
        # 'decoder_input_ids','encoder_outputs', 'past_key_values', 'attention_mask',
    }
    clean_kwargs = {}
    for key in allowed_keys:
        if hasattr(args, key) and getattr(args, key) is not None:
            clean_kwargs[key] = getattr(args, key)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            use_cache=True,
            **clean_kwargs
        )
    if generated_ids.shape[1] > inputs_embeds.shape[1]:
        new_tokens = generated_ids[:, inputs_embeds.shape[1]:]
    else:
        new_tokens = generated_ids

    return new_tokens
    # return torch.cat(generated, dim=1)  # shape: [B, T]
class DataFrame(Dataset):
    def __init__(self, data, args):
        self.source_ids = data['source_ids']
        self.emotion_intensity = data['emotion']
        self.trust_scores = data['trust']
        self.stage_ids = data['stage']
        self.behavior_ids = data['behavior']
        self.strategy_ids = data['strategy_ids']

        # self.knowledge_ids = data['retrieved_knowledge']
        self.response_ids = data['response_ids']
        self.gen_source_ids=data['gen_source_ids']
        # self.instruction_ids=data['instruction_ids']
        self.max_len = args.max_seq_length

    def __getitem__(self, index):
        return self.source_ids[index][:self.max_len], self.strategy_ids[index], self.emotion_intensity[index], \
               self.trust_scores[index], self.stage_ids[index], self.behavior_ids[index], \
                self.response_ids[index],self.gen_source_ids[index],

    def __len__(self):
        return len(self.source_ids)


def collate_fn(data):
    source_ids, strategy_ids,emotion_intensity,trust_ids,stage_ids,behavior_ids,response_ids,gen_source_ids= zip(*data)

    input_ids = [torch.tensor(source_id).long() for source_id in source_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    attention_mask = input_ids.ne(0)
    emotion = torch.tensor(emotion_intensity).long()
    trust = torch.tensor(trust_ids).long()
    stage = torch.tensor(stage_ids).long()
    behavior = torch.tensor(behavior_ids).long()
    input_ids = [torch.tensor(source_id).long() for source_id in source_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=EOS_TOKEN_ID)
    labels = [torch.tensor(response_id[1:] + [EOS_TOKEN_ID]).long() for response_id in response_ids]
    labels = pad_sequence(labels, batch_first=True, padding_value=EOS_TOKEN_ID)
    response_ids = [torch.tensor(response_id).long() for response_id in response_ids]
    response_ids = pad_sequence(response_ids, batch_first=True, padding_value=EOS_TOKEN_ID)
    strategy_ids = torch.tensor(strategy_ids).long()
    gen_input_ids = [torch.tensor(gen_source_id).long() for gen_source_id in gen_source_ids]
    gen_input_ids = pad_sequence(gen_input_ids, batch_first=True, padding_value=EOS_TOKEN_ID)
    gen_attention_mask = gen_input_ids.ne(EOS_TOKEN_ID)

    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'strategy_ids': strategy_ids,
            'emotion': emotion,
            'trust': trust,
            'stage': stage,
            'behavior': behavior,
            # 'knowledge_ids': knowledge_ids,
            # 'instruction': instruction_ids,
            'response_ids': response_ids,
            'labels':labels,
            'gen_input_ids': gen_input_ids,
            'gen_attention_mask':gen_attention_mask,
            # 'instruction_ids':instruction_ids
            }


def test(args, dialogue_planner,prompt_generator, tokenizer,generation_model,generation_tokenizer, embed_model,save_output=False):
    test_dataset = data_reader.load_and_cache_examples(args, tokenizer, generation_tokenizer, evaluate=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, len(args.device_id))

    all_metrics = []
    test_dataloader = DataLoader(DataFrame(test_dataset, args), batch_size=args.eval_batch_size, shuffle=False,
                                 collate_fn=collate_fn)

    # Eval!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(test_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    count = 0
    preds = []
    # response_preds=[]
    targets = []
    scores = []
    sources = []
    responses = []
    golden_responses=[]
    dialogue_planner_to_eval = dialogue_planner.module if hasattr(dialogue_planner, 'module') else dialogue_planner
    dialogue_planner.eval()
    if args.PG==True:
        prompt_generator_to_eval = prompt_generator.module if hasattr(prompt_generator, 'module') else prompt_generator
        prompt_generator.eval()
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        # dialogue planning process

        gen_attention_mask = batch['gen_attention_mask'].to(args.device)
        gen_input_ids = batch['gen_input_ids'].to(args.device)
        conversation = generation_tokenizer.decode(gen_input_ids[0], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
        segments = re.findall(r'(Supporter:|Seeker:)(.*?)(?=Supporter:|Seeker:|$)', conversation, re.DOTALL)

        conversation = [f"{speaker.strip()} {content.strip()}" for speaker, content in segments]
        seq=conversation[-1].strip().lower()
        if seq.startswith("supporter:"):
            continue
        sources.extend([
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in batch['input_ids']
        ])
        inputs = {'input_ids': batch['input_ids'].to(args.device),
                      # 'gen_input_ids': batch['gen_input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'emotion': batch['emotion'].to(args.device),
                      'trust': batch['trust'].to(args.device),
                      'stage': batch['stage'].to(args.device),
                      'behavior': batch['behavior'].to(args.device),
                      'strategy_ids': None,

                      # 'knowledge_ids': batch['knowledge_ids'].to(args.device),
                    }
        with torch.no_grad():
            pred,states= dialogue_planner_to_eval(**inputs)
        scores.extend([p[0] for p in pred.cpu().tolist()])
        pre_strategy_id=pred.argmax(dim=-1).cpu().tolist()
        pre_strategy=ACT[pre_strategy_id[0]]
        preds.extend(pre_strategy_id)
        targets.extend(batch['strategy_ids'].tolist())
        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0)
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        auto_scores = [precision, recall, f1]
        # generation process
        context=" ".join(conversation[-4:])
        if args.RAG == True:
            category = get_strategy_category(pre_strategy)
            if category == "information-related":
                k = search_faiss_and_fetch_text(embed_model, context, args.psyqa_index_path,
                                                 args.psyqa_db_path)
                retrieved_knowledge = category + " knowledge: " + k[0][1]
            elif category == "emotion-related":
                k = search_faiss_and_fetch_text(embed_model, context, args.emotion_index_path,
                                                args.emotion_db_path)
                retrieved_knowledge = category + " knowledge: " + "context " + k[0][0] + " reponse: " + k[0][1]
            elif category == "context-related":
                k = generate_context_knowledge(context, args.chat_ai_model, args.max_new_tokens, args.temperature)
                retrieved_knowledge = category + " knowledge: " + k

            else:
                "Something wrong with category"
        else:
            retrieved_knowledge = None
        # print(retrieved_knowledge, flush=True)

        if args.RAG == True:
            knowledge_ids = generation_tokenizer.encode(retrieved_knowledge)[:args.max_seq_length] + [EOS_TOKEN_ID]
            knowledge_ids = torch.tensor([knowledge_ids]).long().to(args.device)
            with torch.no_grad():
                knowledge_embeddings = generation_model.model.embed_tokens(knowledge_ids)
        else:
            knowledge_embeddings=None
        # instruction_ids = batch['instruction_ids'].to(args.device)
        instruction = ESConvAct[pre_strategy]
        if args.PG==True:
            with torch.no_grad():
                context_embeddings = generation_model.model.embed_tokens(gen_input_ids)  # [1, seq_len, hidden_size]
                instruction_ids = generation_tokenizer.encode(instruction) + [EOS_TOKEN_ID]
                instruction_ids = torch.tensor([instruction_ids]).long().to(args.device)
                instruction_embeddings = generation_model.model.embed_tokens(instruction_ids)
                embedding=prompt_generator_to_eval(context_embeddings= context_embeddings,
                                         attention_mask=gen_attention_mask,
                          knowledge_embeddings=knowledge_embeddings,
                                                   instruction_embeddings=instruction_embeddings,
                                                   train_mode='test'
                          )
                bsz=embedding.size(0)
                First_embedding=generation_model.model.embed_tokens(torch.tensor([BOS_TOKEN_ID]).long().to(args.device))
                embedding=torch.cat([embedding,First_embedding.view(bsz,1,-1)],dim=1)
                response_pred=generate_from_embedding(args,generation_model,embedding)
        else:
            if retrieved_knowledge is not None:
                knowledge_prompt = f"\nNote: Here is some retrieved knowledge that might be helpful:\n{retrieved_knowledge}\n\n"
            else:
                knowledge_prompt = " "
            # context = " ".join([f"{t['speaker']}: {t['text']}" for t in conversation])
            context = " ".join(conversation[-6:])
            messages = [
                    {"role": "system",
                     "content": f"{instruction}"},
                    {"role": "usr",
                     "content": f"{knowledge_prompt} The dialogue context is: {context} Please generate the response directly."},
                ]
            response_pred=generate_response(args,generation_model,generation_tokenizer,messages)
        responses.extend([
            generation_tokenizer.decode(
                response_pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ])
        golden_responses.extend([
            generation_tokenizer.decode(
                r, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for r in batch['response_ids']
        ])
        dist=compute_distinct2(responses)
        bleu=compute_bleu4(responses,golden_responses)
        auto_scores=auto_scores+[dist,bleu]
            # metric_res, auto_scores = metric.compute_metrics([golden_responses],response_preds )
        if save_output:
            REC_PATH ='./results/test_result/' + args.output_dir + '.txt'
            if not os.path.isdir('./results/test_result/' + args.output_dir ):
                os.makedirs('./results/test_result/' + args.output_dir)
            with open(REC_PATH, 'w') as result_file:
                for target, pred, response_pred, golden_resp,source in zip(targets, preds, responses, golden_responses,sources):
                    result_file.write('%s\n\n' % str({'Context': source, 'Pre_Strategy': pred,'True_Strategy': target,'Pre_Response':response_pred,'Golden_Response':golden_resp}))

    logging.info(auto_scores)
    print(auto_scores)
    return auto_scores

def main():
    parser = argparse.ArgumentParser(description="train.py")

    ## Required parameters
    parser.add_argument('--data_name', default='esc', type=str,
                        help="dataset name")
    parser.add_argument('--set_name', default='test', type=str,
                        help="dataset split name")
    parser.add_argument('--model_name', default='roberta', type=str,
                        help="model name")
    parser.add_argument('--generation_model_name', default='llama2', type=str,
                        help="model name")
    parser.add_argument('--model_name_or_path', default="./../roberta-large", type=str,
                        help="model name")
    parser.add_argument('--generation_model_name_or_path', default="./../llama2-7b-chat-hf", type=str,
                        help="model name")
    parser.add_argument("--output_dir", default='test', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", default='./data', type=str,
                        help="The data directory.")
    parser.add_argument("--cache_dir", default='./storage_fast/plm', type=str,
                        help="The cache directory.")
    parser.add_argument("--embed_model_path", type=str,
                        default='all-MiniLM-L6-v2')
    ## Other parameters
    parser.add_argument("--do_train", default=False,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True,
                        help="Whether to run eval.")
    parser.add_argument("--dp_learning_rate", default=2e-6, type=float,#5
                        help="The initial learning rate for Adam.")
    parser.add_argument("--pg_learning_rate", default=6e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--overwrite_output_dir', default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', default=True,
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")
    parser.add_argument('--gpu', default="0", type=str,
                        help="Use CUDA on the device.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="DDP requirement.")
    parser.add_argument('--dp_sft_dir', default='sft/esc/roberta/Truestate_attention/Truerag/best_checkpoint/dialogue_planner',#sft/esc/roberta/Truestate_attention/Truerag/best_checkpoint/dialogue_planner
                        # ../pretrain/outputs/best_pretrain.pt
                        type=str, help="Pretrain model path.")
    parser.add_argument('--pg_sft_dir', default='sft/esc/roberta/Truestate_attention/Falserag/best_checkpoint/prompt_generator',
                        # ../pretrain/outputs/best_pretrain.pt#/scratch-emmy/lustre-emmy-hdd/usr/u15272/LLM/Dialogue_Planning_LLM/tmp/esc/RL-agent/esc-llama2-chat_ai-chat_aipg-epoch-1#sft/esc/roberta/Truestate_attention/Truerag/best_checkpoint/prompt_generator
                        type=str, help="Pretrain model path.")

    parser.add_argument("--chat_ai_model", type=str,
                        default="llama-3.3-70b-instruct")  # qwen2.5-72b-instruct #meta-llama-3.1-70b-instruct #deepseek-r1-distill-llama-70b
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--eos_token_id', type=int, default=2)
    parser.add_argument("--reframing_index_path", type=str, default="./knowledge/reframing_knowledge_faiss.index",
                        help="Path to FAISS index for Reframing")
    parser.add_argument("--reframing_db_path", type=str, default="./knowledge/reframing_knowledge.db",
                        help="CSV file for Reframing knowledge")

    parser.add_argument("--psyqa_index_path", type=str, default="./knowledge/psyqa_knowledge_faiss.index",
                        help="Path to FAISS index for PsyQA")
    parser.add_argument("--psyqa_db_path", type=str, default="./knowledge/psyqa_knowledge.db",
                        help="CSV file for PsyQA knowledge")

    parser.add_argument("--emotion_index_path", type=str, default="./knowledge/emotional_knowledge_faiss.index",
                        help="Path to FAISS index for Emotion-Reflection")
    parser.add_argument("--emotion_db_path", type=str, default="./knowledge/emotional_knowledge.db",
                        help="CSV file for Emotion-Reflection")
    parser.add_argument("--state_attention", default=True, type=bool,
                        help="if add four state aspects")
    parser.add_argument("--PG", default=True, type=bool,
                        help="if add PG to LLM")
    parser.add_argument("--RAG", type=bool, default=False)
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.data_name, args.model_name,str(args.state_attention)+"state_attention",str(args.RAG)+"rag")
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(level=logging.DEBUG, filename=args.output_dir + '/log.txt', filemode='a')

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    device, device_id = utils.set_cuda(args)
    args.device = device
    args.device_id = device_id
    print(args.device,flush=True)
    print(args.device_id, flush=True)
    # Set seed
    utils.set_random_seed(args.seed)

    config = cfg[args.model_name].from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = tok[args.model_name].from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case,
                                                     cache_dir=args.cache_dir)
    generation_config = AutoConfig.from_pretrained(args.generation_model_name_or_path, cache_dir=args.cache_dir)
    generation_tokenizer = AutoTokenizer.from_pretrained(args.generation_model_name_or_path,
                                                         do_lower_case=args.do_lower_case)
    dialogue_planner = MDDP(args, config, tokenizer)
    dialogue_planner.to(args.device)
    generation_model = AutoModelForCausalLM.from_pretrained(args.generation_model_name_or_path)
    generation_model.to(args.device)
    for param in generation_model.parameters():
        param.requires_grad = False
    if args.PG==True:
        prompt_generator = PG(args, generation_config, generation_tokenizer)
        prompt_generator.to(args.device)
    else:
        prompt_generator=None


    logging.info("Evaluation parameters %s", args)
    if args.dp_sft_dir is not None:
        print('Staring loading policy model from {}'.format(args.dp_sft_dir))
        dialogue_planner.load_model(data_name=args.data_name, filename=args.dp_sft_dir)
        dialogue_planner.to(args.device)
        print(next(dialogue_planner.parameters()).device,flush=True)
    if args.pg_sft_dir is not None and args.PG==True:
        print('Staring loading policy model from {}'.format(args.pg_sft_dir))
        prompt_generator.load_model(data_name=args.data_name, filename=args.pg_sft_dir)
        prompt_generator.to(args.device)
        print(next(dialogue_planner.parameters()).device,flush=True)

    if args.RAG == True:
        embed_model = SentenceTransformer(args.embed_model_path)
    else:
        embed_model=None
    args.set_name = 'test'
    test(args, dialogue_planner,prompt_generator, tokenizer,generation_model,generation_tokenizer,embed_model, save_output=True)


if __name__ == "__main__":
    main()