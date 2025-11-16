import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data.prompt import behaviors,stages
from fastchat.model import load_model, get_conversation_template

import openai

from utils import *
from prompt import *
#from unidecode import unidecode
import nltk
import re
import time
from data.chek_data import average_to_5_level
system_role = {'esc':'Supporter', 'cima': 'Teacher', 'cb': 'Buyer'}
user_role = {'esc':'Seeker', 'cima': 'Student', 'cb': 'Seller'}
message_format = {'esc': ESConvMessages}

YOUR_API_KEY = ""



class Env(object):
    def __init__(self, args, dataset, mode, env_model=None, env_tokenizer=None):
        if 'vicuna' in [args.system, args.user, args.critic] or 'llama2' in [args.system, args.user, args.critic]:
            if mode == 'train':
                self.vicuna_model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,             # 自动分配多个GPU
                ).to(args.device)#.eval()
                if args.if_float16==True:
                    self.vicuna_model.to(torch.float16)
                self.vicuna_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                for param in self.vicuna_model.parameters():
                    param.requires_grad = False

            else:
                self.vicuna_model = env_model
                self.vicuna_tokenizer = env_tokenizer
        self.args = args
        self.dataset = dataset[mode]
        self.max_turn = args.max_turn
        self.conversation = []
        self.cur_conver_step = 0
        self.test_num = 0
        self.mode = mode
        self.states=[]
        self.reward_dict = {
            'esc': {
                'worse': -1.0,
                'same': -0.5,
                'better': 0.5,
                'solved': 1.0,
            },
            'cima': {
                'incorrect': -1.0,
                'did not': -0.5,
                'part': 0.5,
                'whole': 1.0,
            },
        }
        self.response_reward_dict = {
                'not': 0,
                'poor': 0.5,
                'good': 1,
        }
        self.past_emotions = []
        self.past_trusts = []
        self.N = 2  # 计算奖励的滑动窗口大小
        set_random_seed(args.seed)

        
    def reset(self):
        self.past_emotions.clear()
        self.past_trusts.clear()
        state=dict()
        self.cur_conver_step = 0
        if self.mode == 'train':
            self.case = np.random.choice(self.dataset)
        elif self.mode == 'test':
            self.case = self.dataset[self.test_num]
            self.test_num += 1
        
        if self.args.data_name == 'esc':
            self.conversation = [{"role":"Seeker", "content":self.case['situation']}]
        elif self.args.data_name == 'cima':
            self.conversation = [{"role":"Teacher", "content":self.case['dialog'][0]['text']}, {"role":"Student", "content":self.case['dialog'][1]['text']}]
        elif self.args.data_name == 'cb':
            self.conversation = [{"role":"Buyer", "content":"Hi, how much is the %s?" % self.case['item_name']}, {"role":"Seller", "content":"Hi, this is a good %s and its price is %s." % (self.case['item_name'], self.case['seller_price'])}]
        print(self.conversation,flush=True)
        state.update({"emotion":5,"behavior":"statement-emotion","stage":"precontemplation","trust":3})
        self.states=[state]
        return self.conversation,self.states


    def step(self, action,retrieved_knowledge,prompt_embedding=None):
        state=dict()
        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))
        print(action,flush=True)

        if prompt_embedding is None:
            messages = message_format[self.args.data_name](self.case, 'system', self.conversation, action,retrieved_knowledge)
            response = self.generate_response(self.args.system, messages, system_role[self.args.data_name])
            response = self.postprocess_response(response, user_role[self.args.data_name])
        else:
            response=self.generate_response_from_query(prompt_embedding)
        self.conversation.append({"role":system_role[self.args.data_name],"content":response})
        print(self.conversation[-1],flush=True)
        # 获取回复的奖励
        if self.mode != 'test':
            messages = message_format[self.args.data_name](self.case, 'response_reward', self.conversation[-6:],action=action)
            response_reward=self.get_response_reward(self.args.chat_ai_model,messages,'response_reward',self.args.critic)
            print(response_reward,flush=True)
        else:
            print("No response reward in test mode",flush=True)
            response_reward = 0
        # self-play user's response
        messages = message_format[self.args.data_name](self.case, 'user', self.conversation)
        user_response = self.generate_response(self.args.user, messages, user_role[self.args.data_name])
        user_response = self.postprocess_response(user_response, system_role[self.args.data_name])
        self.conversation.append({"role": user_role[self.args.data_name], "content": user_response})
        print(self.conversation[-1],flush=True)
        # get different dimensions of seeker's state
        if self.args.state_attention == True:
            state_message = ESConvMessages(self.case, "state_tracking_ebs", self.conversation[-6:])
            state = self.get_user_states(state_message, self.args.chat_ai_model,state, "state_tracking_ebs",
                                         self.args.system)
            state_message = ESConvMessages(self.case, "state_tracking_trust", self.conversation)
            state = self.get_user_states(state_message, self.args.chat_ai_model, state, "state_tracking_trust",
                                         self.args.system)
            self.states.append(state)
            # 输出用户生成的回复以及当前的状态
            print(self.states[-1],flush=True)
        else:
            print("Strategy attention is not enabled, skipping state tracking.",flush=True)

        messages = message_format[self.args.data_name](self.case, 'critic', self.conversation)
        critic_score=self.get_critic(self.args.chat_ai_model,messages,self.args.critic)
        if self.args.data_name == 'esc':
            # if state['emotion']==1 or state['emotion']==2:
            #     done=1
            if critic_score>0.5:
                print('--> Goal completed !')
                done = 1
            else:
                if self.cur_conver_step == self.max_turn - 1:
                    print('--> Maximum number of turns reached !')
                    done = -1
                else:
                    print('--> On-going !')
                
        self.cur_conver_step += 1
        return self.states,self.conversation,response_reward,done

    def postprocess_response(self, response, role):
        #print(response)
        if role in response:
            response = response.split(role)[0].strip()
        sents = nltk.sent_tokenize(response)
        if len(sents) == 1:
            if response[-1] not in ['.','!','?',':']:
                return response + '.'
            return response.strip()
        try:
            if sents[-1].strip()[-1] not in ['.','!','?',':']:
                return ' '.join(sents[:-1]).strip()
            else:
                return response.strip()
        except Exception as e:
            return response.strip()

    def generate_response_from_query(self,inputs_embeds):
        batch_size, seq_len, _ = inputs_embeds.shape
        generated = []

        with torch.no_grad():
            outputs = self.vicuna_model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_logits = logits[:, -1, :] / self.args.temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
        next_token_flat = next_token.view(1)
        generated.append(next_token_flat.item())

        for _ in range(self.args.max_new_tokens - 1):
            with torch.no_grad():
                outputs = self.vicuna_model(
                    input_ids=next_token,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            next_token_logits = logits[:, -1, :] / self.args.temperature

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_flat=next_token.view(1)
            generated.append(next_token_flat.item())

            if self.args.eos_token_id is not None and (next_token_flat == self.args.eos_token_id).all():
                break
        # print(generated,flush=True)
        response=self.vicuna_tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        return response

    def generate_response(self, model, messages, role):
        if self.mode == 'test':
            temperature = 0.7
        else:
            temperature = 0.7
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, role)
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            #print(len(input_ids[0]))
            max_new_tokens = self.args.max_new_tokens
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=max_new_tokens,
                temperature = temperature,
                early_stopping=True
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
        elif model == 'chat_ai':
            messages = chatgpt_prompt(messages, role)
            output = query_chat_ai_model(
                api_key="",
                messages=messages,
                url="",
                model=self.args.chat_ai_model,
                max_tokens=self.args.max_new_tokens,
                temperature=temperature
            )
        return output
    def get_response_reward(self, model_name, messages, role="response_reward", model="chat_ai"):
        if model == 'chat_ai':
            messages = chatgpt_prompt(messages, role)
            outputs = query_chat_ai_model(
                api_key="",
                messages=messages,
                url="",
                model=self.args.chat_ai_model,  # qwen2.5-72b-instruct #meta-llama-3.1-70b-instruct
                max_tokens=16,
                temperature=1.1,
                n=10
            )
            #     convert text to dict
            rewards = []
            for output in outputs:
                for key in self.response_reward_dict:
                    if key in output.lower():
                        rewards.append(self.response_reward_dict[key])
                        break
            if len(rewards) == 0:
                reward = 0
            else:
                reward = sum(rewards) / len(rewards)

        return reward

    def get_user_states(self, messages, model_name,state, role, model="llama2"):
        if model == 'chat_ai':
            messages = chatgpt_prompt(messages, role)
            output = query_chat_ai_model(
                api_key="",
                messages=messages,
                url="",
                model=model_name,
                temperature=0.0

            )
            #     convert text to dict
            result = output.strip("**")
            result = result.strip()
            result = re.split(r':', result)
            print(result,flush=True)
        elif model == "llama2":
            prompt = self.vicuna_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            input_ids=torch.tensor(input_ids).long().to(self.args.device)
            output_ids = self.vicuna_model.generate(
                input_ids,
                # max_new_tokens=200,
                temperature=0.5,
                do_sample=True,
                early_stopping=True,
                # num_return_sequences=10,
            )
            # outputs = []
            # for o in output_ids:
            output_ids = output_ids[0][len(input_ids[0]):]
            output = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                                      spaces_between_special_tokens=False)

        elif model == "vicuna":
            prompt = self.vicuna_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            input_ids = torch.tensor(input_ids).long().to(self.args.device)

            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                temperature=0.5,
                do_sample=True,
                early_stopping=True,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                                      spaces_between_special_tokens=False)
                outputs.append(output)

        if role=="state_tracking_emotion":
            # 支持 ":" 或 "="，可选括号后缀
            emotion_pattern = r'Negative emotional intensity\s*[:=]\s*(\d)'
            try:
                output_parts = output.split("\n\n")
                result = output_parts[-1]
                match = re.search(emotion_pattern, result, re.IGNORECASE)
                if match:
                    emotion = int(match.group(1))
                else:
                    raise ValueError("Primary emotion match failed")
            except:
                fallback_matches = re.findall(emotion_pattern, output, re.IGNORECASE)
                print(fallback_matches,fallback_matches)
                if fallback_matches:
                    emotion = int(fallback_matches[-1])
                    print(f"[Fallback Match] emotion: {emotion}")
                else:
                    emotion = 3  # fallback 默认值，可自定义
            state.update({
                "emotion": emotion,
                })
        elif role=="state_tracking_behavior":
            behavior_pattern = (
                    r'Behavior\s*[:=]\s*(?:\d+\s*[,()]?\s*)?'  # optional number
                    r'(?P<behavior>(' + '|'.join(behaviors) + r'))'
            )

            try:
                output_parts = output.split("\n\n")
                result = output_parts[-1]
                match = re.search(behavior_pattern, result, re.IGNORECASE)
                if match:
                    behavior = match.group("behavior").strip().lower()
                else:
                    raise ValueError("Primary behavior match failed")
            except:
                fallback_matches = re.findall(behavior_pattern, output, re.IGNORECASE)
                if fallback_matches:
                    print(fallback_matches,flush=True)
                    behavior = fallback_matches[-1]
                    if isinstance(behavior, tuple):
                        behavior=behavior[-1].strip().lower()
                    else:
                        behavior = behavior.strip().lower()
                    print(f"[Fallback Match] behavior: {behavior}")
                else:
                    behavior = "others"  # fallback default
                    print("[Warning] No behavior found, using default.")

            state.update({"behavior": behavior})


        elif role == "state_tracking_stage":
            stage_pattern = (
                    r'Stage of Change\s*[:=]\s*'
                    r'(?:\d+\s*[,()]?\s*)?'
                    r'(?P<stage>(' + '|'.join(stages) + r'))'
            )
            try:
                output_parts = output.split("\n\n")
                result = output_parts[-1]
                match = re.search(stage_pattern, result, re.IGNORECASE)
                if match:
                    stage = match.group("stage").strip().lower()
                else:
                    raise ValueError("Primary stage match failed")

            except:
                fallback_matches = re.findall(stage_pattern, output, re.IGNORECASE)
                if fallback_matches:

                    stage = fallback_matches[-1]
                    if isinstance(stage, tuple):
                        stage=stage[-1].strip().lower()
                    else:
                        stage = stage.strip().lower()
                    print(f"[Fallback Match] stage: {stage}")
                else:
                    stage = "contemplation"  # fallback default，可自定义
                    print("[Warning] No stage found, using default.")

            state.update({"stage": stage})
        elif role == "state_tracking_ebs":
            try:
                output_parts = output.split("\n\n")
                result = output_parts[-1]
                output_str = " ".join(output_parts)
                # 正则模式
                emotion_pattern = r'negative emotional intensity\s*[:=]\s*(\d)'
                behavior_pattern = r'behavior\s*[:=]?\s*(?:\d+\s*[,()]?\s*)?(?P<behavior>\b(?:' + '|'.join(
                    behaviors) + r'))'
                stage_pattern = r'stage of change\s*[:=]?\s*(?:\d+\s*[,()]?\s*)?(?P<stage>\b(?:' + '|'.join(
                    stages) + r'))'

                # emotion
                emotion_match = re.search(emotion_pattern, result, re.IGNORECASE)
                emotion = int(emotion_match.group(1)) if emotion_match else None

                # behavior
                behavior_match = re.search(behavior_pattern, result, re.IGNORECASE)
                behavior = behavior_match.group("behavior").strip().lower() if behavior_match else None

                # stage
                stage_match = re.search(stage_pattern, result, re.IGNORECASE)
                stage = stage_match.group("stage").strip().lower() if stage_match else None

                if not all([emotion, behavior, stage]):
                    raise ValueError("Some fields missing")

            except Exception as e:
                print(f"[Fallback triggered due to]: {e}")

                def extract_last_text(matches):
                    if not matches:
                        return None
                    last = matches[-1]
                    if isinstance(last, tuple):
                        return last[-1].strip()
                    return last.strip()
                emo_lst = re.findall(emotion_pattern, output_str, re.IGNORECASE)

                emotion = int(extract_last_text(emo_lst)) if emo_lst else 3
                behavior_lst = re.findall(behavior_pattern, output_str, re.IGNORECASE)
                behavior = extract_last_text(behavior_lst) or "others"
                stage_lst = re.findall(stage_pattern, output_str, re.IGNORECASE)
                stage = extract_last_text(stage_lst) or "precontemplation"

            state.update({
                "emotion": emotion,
                "behavior": behavior.lower(),
                "stage": stage.lower()
            })

        elif role=="state_tracking_trust":
            reliability_pattern = r"reliability\s*[:=]\s*(\d+)"
            response_competence_pattern = r"response competence\s*[:=]\s*(\d+)"
            perceived_understandability_pattern = r"perceived understandability\s*[:=]\s*(\d+)"

            try:
                output_parts = output.split("\n\n")
                result = output_parts[-1]
                reliability_match = re.search(reliability_pattern, result, re.IGNORECASE)
                response_competence_match = re.search(response_competence_pattern, result, re.IGNORECASE)
                perceived_understandability_match = re.search(perceived_understandability_pattern, result,
                                                              re.IGNORECASE)
                if reliability_match and response_competence_match and perceived_understandability_match:
                    reliability = int(reliability_match.group(1))
                    response_competence = int(response_competence_match.group(1))
                    perceived_understandability = int(perceived_understandability_match.group(1))
                else:
                    raise ValueError("Primary trust score matching failed.")
            except:
                print(output_parts,flush=True)
                reliability_list = re.findall(reliability_pattern, output, re.IGNORECASE)
                response_competence_list = re.findall(response_competence_pattern, output, re.IGNORECASE)
                perceived_understandability_list = re.findall(perceived_understandability_pattern, output,
                                                              re.IGNORECASE)

                reliability = int(reliability_list[-1]) if reliability_list else 3
                response_competence = int(response_competence_list[-1]) if response_competence_list else 3
                perceived_understandability = int(
                    perceived_understandability_list[-1]) if perceived_understandability_list else 3

                print(
                    f"[Fallback] reliability: {reliability}, response_competence: {response_competence}, perceived_understandability: {perceived_understandability}")

            trust = average_to_5_level(
                np.mean([reliability, response_competence, perceived_understandability])
            )

            state.update({
                "trust": trust
            })
        else:
            emotion, behavior, stage, reliability, response_competence, perceived_understandability = re.split(r',',
                                                                                                           result[-1])
            trust = average_to_5_level(
                np.mean([int(reliability), int(response_competence), int(perceived_understandability)]))
            state.update( {
                "emotion": int(emotion),
                "behavior": behavior.strip().lower(),
                "stage": stage.strip().lower(),
                "trust": trust
            })
        return state

    def get_critic(self, model_name, messages, model):
        print("Get Critic")
        if model == 'vicuna':
            prompt = vicuna_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'llama2':
            prompt = llama2_prompt(messages, 'critic')
            #print(prompt)
            input_ids = self.vicuna_tokenizer([prompt]).input_ids
            output_ids = self.vicuna_model.generate(
                torch.as_tensor(input_ids).cuda(),
                max_new_tokens=16,
                temperature = 1.1,
                do_sample = True,
                early_stopping=True,
                num_return_sequences=10,
            )
            outputs = []
            for o in output_ids:
                output_id = o[len(input_ids[0]):]
                output = self.vicuna_tokenizer.decode(output_id, skip_special_tokens=True,
                                    spaces_between_special_tokens=False)
                outputs.append(output)
        elif model == 'chatgpt':
            messages = chatgpt_prompt(messages, user_role[self.args.data_name])
            outputs = query_chat_ai_model(
                api_key=YOUR_API_KEY,
                messages=messages,
                model="gpt-3.5-turbo-0613",
                max_tokens=self.args.max_new_tokens,
                temperature=1.1,
                n=10
            )
        elif model == 'chat_ai':
            messages = chatgpt_prompt(messages, user_role[self.args.data_name])
            outputs = query_chat_ai_model(
                api_key="",
                messages=messages,
                url="",
                model=model_name,
                max_tokens=self.args.max_new_tokens,
                temperature=1.1,
                n=10
            )
        print(outputs,flush=True)
        if self.args.data_name in ['esc']:
            rewards = []
            for output in outputs:
                for key in self.reward_dict[self.args.data_name]:
                    if key in output.lower():
                        rewards.append(self.reward_dict[self.args.data_name][key])
                        break
            if len(rewards) == 0:
                reward = 0
            else:
                reward = sum(rewards)/len(rewards)
        return reward


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
                max_tokens=max_tokens,
                n=n,
                #stop=None,
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