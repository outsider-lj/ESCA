import torch
from torch.distributions import Categorical
import random
import numpy as np
from torch.optim import AdamW
from transformers import BertModel, RobertaModel, AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from utils import *
from prompt import ESConvAct,behaviors,stages#, CIMAAct, CBAct
from itertools import chain
import math

model = {'bert': BertModel, 'roberta': RobertaModel}
act = {'esc': ESConvAct}# 'cima': CIMAAct, 'cb': CBAct
TMP_DIR = {
    'esc': './tmp/esc',
    'cima': './tmp/cima',
    'cb': './tmp/cb',
}
class AttentionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states, mask=None):
        # hidden_states: [batch_size, seq_len, hidden_size]
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_probs = self.softmax(attn_weights)

        output = torch.matmul(attn_probs, V)
        pooled = output.mean(dim=1)  # 可替换为 CLS or attention weighted pool
        return pooled
class MLPFeatureExtractor(nn.Module):
    """ 用于提取不同维度（情绪、信任度、行为、改变阶段）的特征 """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPFeatureExtractor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)  # (batch_size, seq_len, output_dim)

class MDDP(nn.Module):
    def __init__(self, args, config, tokenizer):
        super().__init__()
        self.policy = model[args.model_name].from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir)
        if args.state_attention==True:
            self.emotion_embedding=nn.Embedding(5, config.hidden_size)
            self.behavior_embedding = nn.Embedding(9, config.hidden_size)
            self.stage_embedding = nn.Embedding(5, config.hidden_size)
            self.trust_embedding = nn.Embedding(5, config.hidden_size)
            self.proj1=nn.Linear(config.hidden_size,config.hidden_size)
            self.proj2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.proj3 = nn.Linear(config.hidden_size, config.hidden_size)
            self.convert_liner = nn.Linear(config.hidden_size*2, config.hidden_size)
            # self.trust_embedding = nn.Parameter(torch.FloatTensor(1, config.hidden_size))
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size,config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size,config.hidden_size)#4
            )
            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(0.65)
            self.value_function = nn.Linear(config.hidden_size, 1)

        self.act = sorted(list(act[args.data_name].keys()))
        self.classifier = nn.Linear(config.hidden_size, len(self.act))


        self.tokenizer = tokenizer
        self.optimizer = AdamW(
            self.parameters(), lr=args.dp_learning_rate
        )
        self.eps = np.finfo(np.float32).eps.item()
        self.config = config
        self.args = args
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        self.old_probs = []

    def build_input(self, state):
        dial_id = []
        for turn in state[::-1]:
            s = self.tokenizer.encode("%s: %s" % (turn['role'], turn['content']))
            if len(dial_id) + len(s) > self.args.max_seq_length:
                break
            dial_id = s[1:] + dial_id
        inp = s[:1] + dial_id
        return [inp]

    def forward(self, input_ids,attention_mask,emotion,trust,stage,behavior,strategy_ids,train_mode='supervised'):
        outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        if self.args.state_attention==True:
            emo_state = self.emotion_embedding(emotion)
            stage_state = self.stage_embedding(stage)
            behavior_state = self.behavior_embedding(behavior)
            trust_state = self.trust_embedding(trust)
            state_embed = torch.stack([emo_state, trust_state, stage_state, behavior_state], dim=1)  # trust_state,
            state_features= self.mlp(state_embed)
            state_features = self.dropout(state_features)
            context_output = self.proj1(cls_output)
            features = self.proj2(state_features)
            attn_scores = torch.bmm(context_output.unsqueeze(1), features.transpose(-1, -2)) / math.sqrt(
                features.size(-1))
            attn_weights = self.softmax(attn_scores)
            state_features = torch.bmm(attn_weights,
                                       self.proj3(state_features)).squeeze(1) # .unsqueeze(1).transpose(1,2) # (batch_size, num_dimensions, hidden_size)
            combined_output = torch.cat([state_features, cls_output], dim=-1)
            strategy_feature = self.convert_liner(combined_output)
            logits = self.classifier(strategy_feature)
        else:
            strategy_feature=cls_output
            logits = self.classifier(strategy_feature)
        if train_mode == 'supervised':
            return logits,strategy_feature

        elif train_mode == 'rl':
            value = self.value_function(strategy_feature)
            return logits, strategy_feature,value

    def select_action(self, state,emotion,trust,stage,behavior, is_test=False,train_mode='supervised'):
        device=self.args.device
        if self.args.state_attention==True:
            trust = torch.tensor([trust], dtype=torch.long, device=device)
            emotion = torch.tensor([emotion], dtype=torch.long, device=device)
            behavior = torch.tensor([behavior], dtype=torch.long, device=device)
            stage = torch.tensor([stage], dtype=torch.long, device=device)
        inp = self.build_input(state)
        inp = torch.tensor(inp,dtype=torch.long, device=device)
        if train_mode == 'supervised':
            logits,states = self.forward(inp, None, emotion,trust,stage,behavior, None,train_mode='supervised')
            probs = nn.functional.softmax(logits, dim=-1)

        elif train_mode == 'rl':
            logits, states,value = self.forward(inp, None,emotion,trust,stage,behavior,None, train_mode='rl')
            probs = nn.functional.softmax(logits, dim=-1)
            m = Categorical(probs)

        if is_test:
            action = probs.argmax().item()
            return self.act[action]
        else:
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
            self.saved_values.append(value)
            self.old_probs.append(probs.detach())
            return self.act[action],states,value

    def optimize_model(self, train_mode='rl'):
        # 计算折扣奖励 Gt
        device=self.args.device
        print("Rewards:",self.rewards)
        rewards = torch.tensor(self.rewards,device=device)
        returns = torch.zeros_like(rewards,device=device)
        R = 0

        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.args.gamma * R
            returns[t] = R


        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # calculate Advantage（A = Gt - V）
        values = torch.stack(self.saved_values).squeeze()  # 转换成 Tensor
        log_probs = torch.stack(self.saved_log_probs)  # [T]
        old_log_probs = torch.stack(self.old_probs)  # [T]
        advantages = (returns - values).detach()  # 防止梯度回传
        if old_log_probs.size()[0]<1:
            print("No old log probabilities found. Skipping optimization step.")
            return 0.0
        #ppo loss
        policy_losses = []
        value_losses = []
        for log_prob, old_prob, advantage, value,ret in zip(log_probs, old_log_probs, advantages, values,returns):

            new_prob = torch.exp(log_prob)  # 计算新策略的概率
            ratio = new_prob / old_prob

            clipped_ratio = torch.clamp(ratio, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

            value_loss = F.mse_loss(value, ret)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

        self.optimizer.zero_grad()

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        total_loss = policy_loss + value_loss

        total_loss.backward()
        self.optimizer.step()

        cached = torch.cuda.memory_reserved(self.args.device) / 1024 / 1024
        print(cached, flush=True)
        self.rewards.clear()
        self.saved_log_probs.clear()
        self.saved_values.clear()
        self.old_probs.clear()
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.old_probs[:]
        del self.saved_values[:]
        cached = torch.cuda.memory_reserved(self.args.device) / 1024 / 1024
        print(cached, flush=True)

        return total_loss.item()

    def save_model(self, data_name, filename, epoch_user):
            output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model-13.bin'))
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

    def load_model(self, data_name, filename, epoch_user=None):
        if epoch_user:
            output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        else:
            output_dir = filename
        print(output_dir,flush=True)
        if hasattr(self, 'module'):
            self.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            self.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model-13.bin'), map_location=self.args.device))
    def reset(self):
        self.rewards.clear()
        self.saved_log_probs.clear()
        self.saved_values.clear()
        self.old_probs.clear()


class BridgeBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.65):
        super(BridgeBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.kg_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ins_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                                batch_first=True)
        self.feed_forward1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward_query1=nn.Linear(hidden_dim,hidden_dim)
        self.feed_forward2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.feed_forward_query2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,context,knowledge,instruction,query1,query2):
        if knowledge is not None:
            query1_len = query1.size(1)
            query2_len = query2.size(1)
            # Self-attention
            x = torch.cat([query2,query1, context], dim=1)
            self_attn_output, _ = self.self_attn(x, x, x)
            # knowledge
            # Cross-attention with knowledge
            cross_attn_output1, _ = self.kg_cross_attn(self_attn_output, knowledge, knowledge)
            x_kg = self.norm1(x + self.dropout(cross_attn_output1))
            # Feed-forward
            x1 = self.feed_forward1(x_kg)
            x_kg = self.norm_ff(x_kg + self.dropout(x1))
            kg_query = self.feed_forward_query1(x_kg[:, query2_len:query2_len+query1_len, :])
            #instruction
            cross_attn_output2, _ = self.ins_cross_attn(self_attn_output, instruction, instruction)
            x_ins = self.norm2(x + self.dropout(cross_attn_output2))
            x2 = self.feed_forward2(x_ins)
            x_ins = self.norm_ff(x_ins + self.dropout(x2))
            ins_query = self.feed_forward_query2(x_ins[:, :query2_len, :])
        else:
            query2_len = query2.size(1)
            # Self-attention
            x = torch.cat([ query2,context], dim=1)
            self_attn_output, _ = self.self_attn(x, x, x)
            # instruction
            cross_attn_output2, _ = self.ins_cross_attn(self_attn_output, instruction, instruction)
            x_ins = self.norm2(x + self.dropout(cross_attn_output2))
            x2 = self.feed_forward2(x_ins)
            x_ins = self.norm_ff(x_ins + self.dropout(x2))
            ins_query = self.feed_forward_query2(x_ins[:,:query2_len, :])
            kg_query=None
        # ff_history = self.feed_forward_history(cross_attn_output[:, query_len:, :])
        return kg_query, ins_query

class PG(nn.Module):
    def __init__(self, args, config, tokenizer):
        super().__init__()
        self.config = config
        # self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.kg_layers = nn.ModuleList([
            BridgeBlock(config.hidden_size, config.num_attention_heads, 0.65)
            for _ in range(1)
        ])

        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.query1 = nn.Parameter(torch.randn(1, 32, config.hidden_size))  # 可扩展到不同 batch
        self.query2 = nn.Parameter(torch.randn(1, 4, config.hidden_size))
        self.optimizer = AdamW(self.parameters(), lr=args.pg_learning_rate)
        self.log_std = nn.Parameter(torch.zeros(config.hidden_size))  # 固定的标准差
        self.log_std_layer= nn.Linear(config.hidden_size, config.hidden_size)  # 每个 token 的标准差
        self.value_function = nn.Linear(config.hidden_size * 2, 1)
        self.eps = np.finfo(np.float32).eps.item()
        self.config = config
        self.args = args
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        self.old_probs = []

    def reset(self):
        self.rewards.clear()
        self.saved_log_probs.clear()
        self.saved_values.clear()
        self.old_probs.clear()

    def build_input(self, external_knowledge, instruction, dialog_context):
        """
        conduct the input of context, knowledge and instruction
        """
        context_ids = self.tokenizer.encode(dialog_context, return_tensors='pt')
        knowledge_ids = self.tokenizer.encode(external_knowledge, return_tensors='pt')
        instruction_ids = self.tokenizer.encode(instruction, return_tensors='pt')
        return context_ids, knowledge_ids, instruction_ids

    def select_knowledge(self, policy, external_knowledge):
        """
        choose knowledge。
        """
        selected_knowledge = external_knowledge.get(policy, "")
        return selected_knowledge

    def forward(self,context_embeddings,attention_mask, knowledge_embeddings,instruction_embeddings,train_mode="sft"):

        bsz=context_embeddings.size()[0]
        if self.args.RAG==True:
            knowledge_query= self.query1.repeat(bsz, 1, 1)
        else:
            knowledge_query=None
            knowledge_embeddings = None
        instruction_query = self.query2.repeat(bsz, 1, 1)
        for layer in self.kg_layers:
            knowledge_query,instruction_query = layer(context_embeddings, knowledge_embeddings,instruction_embeddings,knowledge_query,instruction_query)
        if self.args.RAG==True:
            soft_prompt=torch.cat([instruction_query,knowledge_query,context_embeddings],dim=1)
        else:
            soft_prompt = torch.cat([ instruction_query,context_embeddings], dim=1)
        if train_mode=="rl":
            # Project each token independently
            soft_prompt_original = torch.cat([instruction_query, knowledge_query], dim=1)
            L_total = soft_prompt_original.size(1)

            # --- Gaussian policy for sequence ---
            std = torch.exp(self.log_std).unsqueeze(0).unsqueeze(0)  # [1, 1, H]
            std = std.expand(bsz, L_total, -1)  # [B, L, H]

            dist = torch.distributions.Normal(soft_prompt_original, std)
            soft_prompt_re = dist.rsample()  # [B, L, H]
            log_prob = dist.log_prob(soft_prompt_re).sum(dim=[1, 2])  # [B], sum over tokens & dim
            prob = torch.exp(log_prob)
            # --- Value function ---
            instruction_pooled = instruction_query.mean(dim=1)
            knowledge_pooled = knowledge_query.mean(dim=1)# [B, H]
            value = self.value_function(torch.cat([instruction_pooled, knowledge_pooled], dim=-1)).squeeze(-1)  # [B]
            # Save for PPO
            self.saved_log_probs.append(log_prob)
            self.saved_values.append(value)
            self.old_probs.append(prob.detach())
            soft_prompt=torch.cat([soft_prompt_re,context_embeddings],dim=1)
            return soft_prompt, log_prob, value
        elif train_mode == "dpg":
            instruction_pooled = instruction_query.mean(dim=1)
            knowledge_pooled = knowledge_query.mean(dim=1)  # [B, H]
            value = self.value_function(torch.cat([instruction_pooled, knowledge_pooled], dim=-1)).squeeze(-1)  # [B]
            self.saved_values.append(value)
            return soft_prompt, None, value
        else:
            return soft_prompt

    def optimize_PPO(self):
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.args.device)
        returns = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.args.gamma * R
            returns[t] = R

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        values = torch.stack(self.saved_values).squeeze()  # [T]
        log_probs = torch.stack(self.saved_log_probs)  # [T]
        old_log_probs = torch.stack(self.old_probs)  # [T]

        advantages = (returns - values).detach()

        gen_policy_losses = []
        value_losses = []
        if old_log_probs.size()[0]<1:
            print("No old log probabilities found. Skipping optimization step.")
            return 0.0
        for log_prob, old_log_prob, advantage, value, ret in zip(log_probs, old_log_probs, advantages, values, returns):
            ratio = torch.exp(log_prob - old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
            value_loss = F.mse_loss(value, ret)
            gen_policy_losses.append(policy_loss)
            value_losses.append(value_loss)

        self.optimizer.zero_grad()
        total_loss = torch.stack(gen_policy_losses).mean() + torch.stack(value_losses).mean()
        total_loss.backward()
        self.optimizer.step()
        cached = torch.cuda.memory_reserved(self.args.device) / 1024 / 1024
        print(cached, flush=True)
        self.rewards.clear()
        self.saved_log_probs.clear()
        self.saved_values.clear()
        self.old_probs.clear()
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.old_probs[:]
        del self.saved_values[:]
        cached = torch.cuda.memory_reserved(self.args.device) / 1024 / 1024
        print(cached, flush=True)
        return total_loss.item()

    def optimize_model(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        if rewards.shape[0] > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        return policy_loss.data

    def save_model(self, data_name, filename, epoch_user):
        output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

    def load_model(self, data_name, filename, epoch_user=None):
        if epoch_user:
            output_dir = TMP_DIR[data_name] + '/RL-agent/' + filename + '-epoch-{}'.format(epoch_user)
        else:
            output_dir = filename
        if hasattr(self, 'module'):
            self.module.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            self.load_state_dict(
                torch.load(os.path.join(output_dir, 'pytorch_model-13.bin'), map_location=self.args.device))#-ep5

