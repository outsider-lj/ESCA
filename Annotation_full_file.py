import json
import argparse
from openai import OpenAI
import re
import numpy as np
from prompt import ESConvMessages
import time
import openai

behaviors = ["greeting", "question", "statement-fact", "statement-opinion", "statement-emotion",
             "command", "feedback, "acknowledgement",  "others"]
stages = ["precontemplation", "contemplation", "preparation", "action", "maintenance"]
def split_json_file(json_file, num_parts=20):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunk_size = len(data) // num_parts

    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = None if i == num_parts - 1 else (i + 1) * chunk_size
        chunk = data[start_idx:end_idx]

        with open(f"./annotated_data/split_part_{i}.json", "w", encoding="utf-8") as f:
            json.dump(chunk, f)

def merge_json_files(output_json, num_parts=10):
    merged_data = []
    for i in range(num_parts):
        json_file = f"_processed_part_{i}.json"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)  # 支持单个字典的情况

    with open(output_json, "w", encoding="utf-8") as out_f:
        json.dump(merged_data, out_f, ensure_ascii=False, indent=2)

def split_conversation(conversation, type, window_size=8, step=4):
    """
    使用滑动窗口切分对话，每 10 句 seeker 话语形成一个子对话，步长为 5。

    :param conversation: List[Dict]，原始对话列表，每个元素为 {'speaker': 'seeker'/'supporter', 'text': '对话内容'}
    :param window_size: int，窗口大小，即每个子对话包含的 seeker 话语数量
    :param step: int，步长，每次滑动窗口时移出的 seeker 话语数量
    :return: List[List[Dict]]，返回切分后的多个子对话
    """
    # 找出所有 seeker 话语的索引
    seeker_indices = [i for i, turn in enumerate(conversation) if turn["speaker"] == "seeker"]

    # 记录最终的分段对话
    split_dialogues = []
    turns_of_seeker = []
    start = 0  # 滑动窗口起始索引
    while start < len(seeker_indices):
        end = min(start + window_size, len(seeker_indices))  # 计算窗口终点

        # 获取 seeker 窗口的索引
        seeker_window = seeker_indices[start:end]

        # 确保选取完整的对话窗口（包括 supporter）
        min_index = seeker_window[0]  # seeker 窗口起点的索引
        max_index = seeker_window[-1]  # seeker 窗口终点的索引

        # 向前扩展以包含 supporter 话语
        while min_index > 0 and conversation[min_index - 1]["speaker"] == "supporter":
            min_index -= 1

        # 向后扩展以包含 supporter 话语
        while max_index < len(conversation) - 1 and conversation[max_index + 1]["speaker"] == "supporter":
            max_index += 1
        context = ''
        # last_speaker = ''
        i = 0
        for utterance in conversation[min_index:max_index + 1]:
            speaker = utterance['speaker']
            cleaned_text = utterance['text'].replace('\n', '')
            text_line = speaker + ': ' + cleaned_text
            # 如果是seeker且有情感强度，将其添加到同一行
            if speaker == 'seeker' and type == "emotion" and utterance["emotion_intensity"] is not None:
                text_line = text_line + " (Current emotion intensity is " + utterance['emotion_intensity'] + ')'
            context = context + text_line + '\n'  # 每个话语单独一行
            if speaker == 'seeker':
                i += 1
        # 取出完整的对话窗口
        split_dialogues.append(context)
        turns_of_seeker.append(i)
        # 结束条件：如果已经包含所有 seeker 话语，则跳出
        if end == len(seeker_indices):
            break
        # 滑动窗口
        start += step
    return split_dialogues, turns_of_seeker


def extract_emo_intensity(output):
    emo_intensity = []
    results = re.split(r'\n', output)
    for r in results:
        print(f"Processing result: {r}")
        # 跳过空行
        if not r.strip():
            continue

        number_match = re.search(r':\s*([1-5])', r)

        if number_match:
            emo_intensity.append(number_match.group(1))
            continue

        # 从括号中提取
        # number_match = re.search(r'\((\d+)\):\s*([1-5])', r)
        # if number_match:
        #     emo_intensity.append(number_match.group(2))
        #     continue

        # 查找任何1-5的数字
        number_match = re.search(r'[1-5]', r)
        if number_match:
            emo_intensity.append(number_match.group(0))
            continue

        print(f"无法从结果中提取情感强度值: {r}")

    print(f"提取的情感强度值: {emo_intensity}")
    return emo_intensity


def extract_trust_score(outputs):
    reliability_scores = []
    response_competence_scores = []
    perceived_understandability_scores = []
    results = re.split(r'\n', outputs)
    for result in results:
        # print(f"Processing result: {result}")  # 打印每一行结果
        result = result.strip("**")
        result = result.strip()
        result = re.split(r':', result)
        if len(result) > 1:
            b_s = result[-1].strip()
            print(f"trust: {b_s}" + f"Processing result: {result}")  # 打印行为和阶段字符串
            parts = re.split(r',', b_s)
            if len(parts) == 3:
                r, c, u = parts
                r = r.strip()
                c = c.strip()
                u = u.strip()
                if re.match(r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$', r):
                    reliability_scores.append(r)
                if re.match(r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$', c):
                    response_competence_scores.append(c)
                if re.match(r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$', u):
                    perceived_understandability_scores.append(u)
            else:
                print(f"Unexpected format: {b_s}")
        else:
            print(f"Unexpected format: {result}")
    return reliability_scores, response_competence_scores, perceived_understandability_scores


def extract_behavior_and_stage(outputs):
    one_behaviors = []
    one_stages = []
    # results = re.split(r'\n', outputs)  # \d+\.\s*
    # print(f"完整API响应: {outputs}")  # 打印完整响应
    pattern = re.compile(r'\((\d+)\):\s*(.+)')
    temp = {}

    for match in pattern.finditer(outputs):
        num = match.group(1)
        label = match.group(2).strip()
        temp[num] = label  # 后出现的会覆盖前面的

    # 将字典转换为排序后的列表形式
    results = [f"{num}: {label}" for num, label in sorted(temp.items(), key=lambda x: int(x[0]))]
    for result in results:
        if not result.strip():  # 跳过空行
            continue
        # print(f"Processing result: {result}")  # 打印每一行结果
        result = result.strip("**")
        result = result.strip()
        result = re.split(r':', result)
        if len(result) > 1:
            b_s = result[-1].strip()
            # print(f"Behavior and stage string: {b_s}")  # 打印行为和阶段字符串
            parts = re.split(r',', b_s)
            if len(parts) >= 2:
                b = parts[-2].strip()
                if b == "acknowledgment":
                    b = "acknowledgement"
                s = parts[-1].strip()
                one_behaviors.append(b)
                one_stages.append(s)
                if b not in behaviors or s not in stages:
                    print(f"content>2: {b},{s}; Processing result: {result}")
            else:
                try:
                    b, s = re.split(r',', result[-1])
                    b = b.strip()
                    s = s.strip()
                except:
                    print(result,flush=True)
                one_behaviors.append(b)
                one_stages.append(s)
                if b not in behaviors or s not in stages:
                    print(f"content: {b},{s}; Processing result: {result}")
                print(f"content: {b},{s}")
        else:
            print(f"Unexpected format: {result}")

    print(f"提取的行为: {one_behaviors}")
    print(f"提取的阶段: {one_stages}")

    return one_behaviors, one_stages


def extract_behavior(outputs):
    one_behaviors = []
    results = re.split(r'\n', outputs)  # \d+\.\s*
    # print(f"behavior完整API响应: {outputs}")  # 打印完整响应
    for result in results:
        if not result.strip():  # 跳过空行
            continue
        # print(f"Processing result: {result}")  # 打印每一行结果
        result = result.strip("**")
        result = result.strip()
        # 尝试提取行为信息
        # 方法1：从格式化输出中提取
        behavior_match = re.search(r'\((\d+)\):\s*(\w+)', result)
        if behavior_match:
            behavior = behavior_match.group(2).lower()
            if behavior in behaviors:
                one_behaviors.append(behavior)
                continue

        # 方法2：从冒号分隔的格式中提取
        parts = re.split(r':', result)
        if len(parts) > 1:
            behavior_part = parts[-1].strip()
            # 检查是否包含逗号（可能是行为,阶段格式）
            if ',' in behavior_part:
                behavior = behavior_part.split(',')[0].strip().lower()
            else:
                behavior = behavior_part.lower()

            if behavior in behaviors:
                one_behaviors.append(behavior)
                continue
        # 方法3：直接在文本中搜索行为关键词
        for behavior in behaviors:
            if behavior.lower() in result.lower():
                one_behaviors.append(behavior)
                break
        else:
            print(f"无法从结果中提取behavior值: {result}")
            break
    print(f"提取的behavior值: {one_behaviors}")
    return one_behaviors


def extract_stage(outputs):
    one_stages = []
    results = re.split(r'\n', outputs)  # \d+\.\s*
    # print(f"stage完整API响应: {outputs}")  # 打印完整响应
    for result in results:
        # print(f"Processing result: {result}")  # 打印每一行结果
        if not result.strip():  # 跳过空行
            continue
        result = result.strip("**")
        result = result.strip()

        # 尝试提取阶段信息
        # 方法1：从格式化输出中提取
        stage_match = re.search(r'\((\d+)\):\s*(\w+)', result)
        if stage_match:
            stage = stage_match.group(2).lower()
            if stage in stages:
                one_stages.append(stage)
                continue

        # 方法2：从冒号分隔的格式中提取
        parts = re.split(r':', result)
        if len(parts) > 1:
            stage_part = parts[-1].strip()
            # 检查是否包含逗号（可能是行为,阶段格式）
            if ',' in stage_part:
                stage = stage_part.split(',')[-1].strip().lower()
            else:
                stage = stage_part.lower()

            if stage in stages:
                one_stages.append(stage)
                continue

        # 方法3：直接在文本中搜索阶段关键词
        for stage in stages:
            if stage.lower() in result.lower():
                one_stages.append(stage)
                break
        else:
            print(f"无法从结果中提取stage值: {result}")
            break

    print(f"提取的stage值: {one_stages}")
    return one_stages


def supplyment(type, d, scores):
    seeker_num = 0
    if "emotion" in type:
        for utt in d["dialog"]:
            if utt["speaker"] == "seeker":
                utt["emotion_intensity"] = scores[seeker_num]
                seeker_num = seeker_num + 1
    if "trust" in type:
        reliability, response_competence, perceived_understandability = scores
        for utt in d["dialog"]:
            if utt["speaker"] == "seeker":
                utt["reliability"] = reliability[seeker_num]
                utt["response_competence"] = response_competence[seeker_num]
                utt["perceived_understandability"] = perceived_understandability[seeker_num]
                utt["trust"] = np.mean([int(reliability[seeker_num]), int(response_competence[seeker_num]),
                                        int(perceived_understandability[seeker_num])])
                seeker_num = seeker_num + 1
    if "behavior" in type and "change" in type:
        behaviors, stages = scores
        for utt in d["dialog"]:
            if utt["speaker"] == "seeker":
                utt["behavior"] = behaviors[seeker_num]
                utt["stage_of_change"] = stages[seeker_num]
                seeker_num = seeker_num + 1
    if "behavior" in type and "change" not in type:
        behaviors = scores
        for utt in d["dialog"]:
            if utt["speaker"] == "seeker":
                utt["behavior"] = behaviors[seeker_num]
                # utt["stage_of_change"] = stages[seeker_num]
                seeker_num = seeker_num + 1
    if "stage" in type and "change" not in type:
        stages = scores
        for utt in d["dialog"]:
            if utt["speaker"] == "seeker":
                # utt["behavior"]=behaviors[seeker_num]
                utt["stage_of_change"] = stages[seeker_num]
                seeker_num = seeker_num + 1


def query_chat_ai_model(api_key: str, messages: str, url: str, model: str = "meta-llama-3.1-70b-intruct",
                        max_tokens: int = 128, temperature: float = 0.0, wait_time: int = 5):
    client = OpenAI(
        api_key=api_key,
        base_url=url
    )
    while True:
        try:
            completions = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return completions.choices[0].message.content.strip()
        except openai.InternalServerError:
            print("Something went wrong with the server, retrying in", wait_time, "seconds...")
            time.sleep(wait_time)


def process_data(d, model, annotation_type, initial_window, initial_step):
    w = initial_window
    s = initial_step
    max_attempts = 5  # 最大尝试次数，避免无限循环
    attempt = 0

    # 首先检查seeker话语总数
    seeker_count = sum(1 for utt in d["dialog"] if utt["speaker"] == "seeker")

    # 如果seeker话语数量少于窗口大小，直接使用实际数量作为窗口大小
    if seeker_count <= w:
        w = seeker_count
        s = seeker_count  # 步长设为窗口大小，确保只处理一次

    while attempt < max_attempts:
        input_data, turns_of_seeker = split_conversation(d['dialog'], annotation_type, w, s)
        outputs = []
        for input, l in zip(input_data, turns_of_seeker):
            # print(f"{input}")
            message = ESConvMessages(args.annotation_type, l, input)
            response = query_chat_ai_model(
                api_key="",
                messages=message,
                url="",
                model=model,
                temperature=0,
                wait_time=args.wait_time
            )
            time.sleep(1)
            if "deepseek" in args.chat_ai_model:
                response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
                response = response.strip("\n+")
            if "emotion" in args.annotation_type:
                a = extract_emo_intensity(response)
                if len(a) == l:
                    outputs.append(a)
                    # 如果seeker话语数量少于等于窗口大小，且已成功提取，直接跳出循环
                    if seeker_count <= initial_window:
                        break
                else:
                    print("turns of seeker are wrong.", flush=True)
                    break
            elif "trust" in args.annotation_type:
                a, b, c = extract_trust_score(response)
                if len(a) == l:
                    outputs.append([a, b, c])
                else:
                    print("turns of seeker are wrong.", flush=True)
                    break
            elif "change" in args.annotation_type and "behavior" in args.annotation_type:
                a, b = extract_behavior_and_stage(response)
                if len(a) == l:
                    outputs.append([a, b])
                else:
                    print("turns of seeker are wrong.", flush=True)
                    break
            elif "stage" in args.annotation_type and "change" not in args.annotation_type:
                a = extract_stage(response)
                if len(a) == l:
                    outputs.append(a)
                else:
                    print("turns of seeker are wrong.", flush=True)
                    break
            elif "behavior" in args.annotation_type and "change" not in args.annotation_type:
                a = extract_behavior(response)
                if len(a) == l:
                    outputs.append(a)
                else:
                    print("turns of seeker are wrong.", flush=True)
                    break
        if len(outputs) == len(turns_of_seeker):
            if "emotion" in args.annotation_type:
                emo_intensity = []
                for i, one_emo_intensity in enumerate(outputs):
                    if i != 0:
                        one_emo_intensity = one_emo_intensity[w - s:]
                    emo_intensity = emo_intensity + one_emo_intensity
                    # print(emo_intensity)
                # 新加内容
                # 检查是否有足够的情感强度值
                seeker_count = sum(1 for utt in d["dialog"] if utt["speaker"] == "seeker")
                if len(emo_intensity) < seeker_count:
                    print(f"出现null值: 预期 {seeker_count} 个情感强度值，但只提取到 {len(emo_intensity)} 个")
                    # 填充缺失的值为默认值"3"
                    emo_intensity.extend(["2"] * (seeker_count - len(emo_intensity)))
                supplyment(args.annotation_type, d, emo_intensity)
            elif "trust" in args.annotation_type:
                reliability_scores = []
                response_competence_scores = []
                perceived_understandability_scores = []
                for i, score in enumerate(outputs):
                    one_scores_r = score[0]
                    one_scores_c = score[1]
                    one_scores_u = score[2]
                    if i != 0:
                        one_scores_r = one_scores_r[w - s:]
                        one_scores_c = one_scores_c[w - s:]
                        one_scores_u = one_scores_u[w - s:]
                    reliability_scores = reliability_scores + one_scores_r
                    response_competence_scores = response_competence_scores + one_scores_c
                    perceived_understandability_scores = perceived_understandability_scores + one_scores_u
                trust_scores = (reliability_scores, response_competence_scores, perceived_understandability_scores)
                supplyment(args.annotation_type, d, trust_scores)
            elif "change" in args.annotation_type and "behavior" in args.annotation_type:
                stage_of_changes = []
                behaviors = []
                for i, score in enumerate(outputs):
                    one_b = score[0]
                    one_c = score[1]
                    if i != 0:
                        one_b = one_b[w - s:]
                        one_c = one_c[w - s:]

                    stage_of_changes = stage_of_changes + one_c
                    behaviors = behaviors + one_b
                behavior_and_changes = (behaviors, stage_of_changes)
                supplyment(args.annotation_type, d, behavior_and_changes)
            elif "stage" in args.annotation_type and "change" not in args.annotation_type:
                stage_of_changes = []
                for i, score in enumerate(outputs):
                    one_c = score
                    if i != 0:
                        one_c = one_c[w - s:]
                    stage_of_changes = stage_of_changes + one_c
                supplyment(args.annotation_type, d, stage_of_changes)
            elif "behavior" in args.annotation_type and "change" not in args.annotation_type:
                behaviors = []
                for i, score in enumerate(outputs):
                    one_b = score
                    if i != 0:
                        one_b = one_b[w - s:]
                    behaviors = behaviors + one_b
                supplyment(args.annotation_type, d, behaviors)
            return d
        w = max(w - 2, 2)  # 窗口大小最小为 2
        s = max(s - 1, 1)  # 步长最小为 1
        print(f"reset window to{w},reset step to {s}", flush=True)
        attempt += 1
    return None  # 返回 None 表示失败
def process_file(paras,input_file,output_file):
    data = json.load(open(input_file))
    # 将一个人连续的话语转化为一个话语
    inputs = []
    window = 4
    step = 2
    # 设置窗口和步伐，步伐=窗口-2，步伐>=窗口/2
    print("Begin Processing", flush=True)
    for index, d in enumerate(data):
        print(index, flush=True)
        process_data(d, paras.chat_ai_model, paras.annotation_type, window, step)
    process_data(d, paras.chat_ai_model, paras.annotation_type, window, step)
    # with open(f'./annotated_data/new/emo_demo30_win6_one_deepseek.json', 'w') as f: #stage_and_behavior
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(input_file, flush=True)
    print("processing is over", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--wait_time', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--annotation_type', type=str, default="trust")  # stage of change and behavior
    parser.add_argument('--model_name', type=str, default="chat_ai")
    parser.add_argument('--chat_ai_model', type=str,
                        default="qwen2.5-72b-instruct")  # qwen2.5-72b-instruct #meta-llama-3.1-70b-instruct #deepseek-r1-distill-llama-70b
    parser.add_argument('--model_path', type=str, default="./../vicuna-7b-v1.5")
    parser.add_argument('--data_file', type=str, default="./original_data/data_demo_30.json")
    parser.add_argument('--output_file', type=str, default="./annotated_data/emo_demo30_win4_llama.json")
    # add_model_args(parser)
    args = parser.parse_args()


    split_json_file("./original_data/ESConv.json", 10)
    for i in range(10):
        process_file(args,f"./annotated_data/deepseek_processed_part_{i}.json", f"./annotated_data/deepseek_processed_part_{i}.json")
    # merge_json_files("./annotated_data/annotated_esconv_full.json")