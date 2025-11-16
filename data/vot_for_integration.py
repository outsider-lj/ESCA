from collections import Counter
import openai  # 需要安装 openai 库
from openai import OpenAI
from prompt import Emotion_intensity,Reliability,Response_Competence,Perceived_Understandability,behavior_rules,stage_of_change_rules,behaviors,stages
import json
import re
import time
import numpy as np
# 设置 OpenAI API 密钥
api_key = "",
url = "",


def vote_intensity(intensity_list):
    """
    对三个情感强度值进行投票，返回最终的情感强度值。
    如果三个值都不同，返回None表示需要第四个人标注。
    """
    counter = Counter(intensity_list)
    if len(counter) == 1:
        return intensity_list[0]
    elif len(counter) == 2:
        # 如果有两个相同的值，返回出现次数最多的值
        return counter.most_common(1)[0][0]
    else:
        # 三个值都不同，返回None
        return None

def get_context(dialog, current_index):
    """
    获取当前轮的上下文（前三轮和后一轮的对话）。
    """
    context = []
    start_index = max(0, current_index - 3)  # 前三轮
    end_index = min(len(dialog), current_index+1 )  # 后一轮

    for i in range(start_index, end_index):
        context.append(f"{dialog[i]['speaker']}: {dialog[i]['text']}")
    return " ".join(context)

def get_fourth_annotation(context, current_text,type):
    """
    调用 LLM API 获取第四个人的情感强度评分。
    输入上下文和当前轮文本，输出情感强度评分。
    """
    if type=="emotion_intensity":
        messages = [
            {"role": "system",
             "content": "You are an AI assistant that annotates emotion intensity in emotional support conversations. "
                        f"You must follow this rule: {Emotion_intensity} to annotate the emotional intensity of the last seeker's utterance"
                        "Output format: The content of the last seeker's utterance: emotion intensity"
                        "Provide only the annotation without explanation."},
            {"role": "user",
             "content": f"You need to annotate the emotional intensity following the example:"
                         "Seeker: I feel like everything I do lately is wrong, and I'm feeling really down.\n"
                        "Supporter: It sounds like you're feeling really frustrated. Have you tried doing anything to relax, like taking a walk or doing something you enjoy?\n"
                        "Seeker: I'm not sure. Maybe I’ve tried a little, but it didn’t really help much.\n"
                        "The result is I'm not sure. Maybe I’ve tried a little, but it didn’t really help much.: 3"
                        f"Now you need to annotate the emotion intensity of {current_text} based on the conversation context {context}"
                        "The answer is:\n"}
        ]
    elif type=="reliability":
        messages = [
            {"role": "system",
             "content": "You are an AI assistant that annotates the score of supporter's reliability in emotional support conversations. "
                        "Reliability of the supporter measures the consistency, honesty, and dependability of the supporter as perceived by the seeker."
                        f"You must follow the rule {Reliability} to annotate the reliability of the last supporter's utterance"
                         "Output format: The content of the last seeker's utterance: reliability score"
                        "Provide only the annotation without explanation."},
            {"role": "user",
             "content": "You need to annotate the reliability score following the example:"
                        "Seeker: I feel like everything I do lately is wrong, and I'm feeling really down.\n"
                    "Supporter: It sounds like you're feeling really frustrated. Have you tried doing anything to relax, like taking a walk or doing something you enjoy?\n"
                    "Seeker: I'm not sure. Maybe I’ve tried a little, but it didn’t really help much.\n\n"
                    "The result is:  I'm not sure. Maybe I’ve tried a little, but it didn’t really help much.: 4"
                        f"Now you need to annotate the reliability of the supporter's utterances before {current_text} based on the conversation context {context}"
                        "Important rules:1.If no prior supporter response exists, initialize Reliability to 3.\n"
                    "2.A score of 5 should only be given for responses that are highly insightful, precise, and perfectly understood. "
                        "The answer is:\n"}
            ]
    elif type == "response_competence":
        messages = [
            {"role": "system",
             "content": "You are an AI assistant that annotates response competence of the supporter in emotional support conversations. "
                        "Response competence of the supporter assesses whether the supporter provides effective, relevant, and competent responses to the seeker’s concerns.\n"
                        f"You must follow the rule {Response_Competence} to annotate the response competence of the last supporter's utterance"
                         "Output format: The content of the last seeker's utterance: score of response competence"
                        "Provide only the annotation without explanation."},
            {"role": "user",
             "content": "You need to annotate the response competence score following the example:"
                        "Seeker: I feel like everything I do lately is wrong, and I'm feeling really down.\n"
                    "Supporter: It sounds like you're feeling really frustrated. Have you tried doing anything to relax, like taking a walk or doing something you enjoy?\n"
                    "Seeker: I'm not sure. Maybe I’ve tried a little, but it didn’t really help much.\n\n"
                    "The result is:  I'm not sure. Maybe I’ve tried a little, but it didn’t really help much.: 4"
                        f"Now you need to annotate the reliability of the supporter's utterances before {current_text} based on the conversation context {context}"
                        "Important rules:1.If no prior supporter response exists, initialize Response Competence to 3.\n"
                    "2.A score of 5 should only be given for responses that are highly insightful, precise, and perfectly understood. "
                        "The answer is:\n"}
        ]
    elif type == "perceived_understandability":
        messages = [
            {"role": "system",
             "content": "You are an AI assistant that annotates perceived understandability of the seeker in emotional support conversations. "
                        "Perceived understandability of the seeker measures how well the seeker feels they understand the supporter’s advice and how to act upon it.\n"
                        f"You must follow the rule {Perceived_Understandability} to annotate the perceived understandability of the seeker"
                         "Output format: The content of the last seeker's utterance: score of perceived understandability"
                        "Provide only the annotation without explanation."},
            {"role": "user",
             "content": f"You must follow this rule: {Perceived_Understandability} to annotate the perceived understandability of the seeker's utterance {current_text} based on the conversation context {context}\n\n"
                        "The output format: The content of current text: score of perceived understandability. for example: I feel like everything I do lately is wrong, and I'm feeling really down.: 3\n\n"
                        "Seeker: I feel like everything I do lately is wrong, and I'm feeling really down.\n"
                    "Supporter: It sounds like you're feeling really frustrated. Have you tried doing anything to relax, like taking a walk or doing something you enjoy?\n"
                    "Seeker: I'm not sure. Maybe I’ve tried a little, but it didn’t really help much.\n"
                    "The result is:  I'm not sure. Maybe I’ve tried a little, but it didn’t really help much.: 3"
                        "Important rules:3.If no prior supporter response exists, initialize Perceived Understandability to 3.\n"
                    "4.A score of 5 should only be given for responses that are highly insightful, precise, and perfectly understood. "
                        "The answer is:\n"}
        ]
    elif type == "behavior":
        messages = [
            {"role": "system",
             "content": "You are an AI assistant that annotates dialogue behavior in emotional support conversations. "
                        f"You must follow this rule: {behavior_rules} to annotate the dialogue behavior of the last seeker's utterance"
                        "Output format: The content of the last seeker's utterance: <dialogue behavior>"
                        "Provide only the annotation without explanation."},
            {"role": "user",
             "content": f"You need to annotate the dialogue behavior following the example:"
                       "Seeker: Lately, I’ve been feeling so exhausted. No matter how much I sleep, I still wake up tired.\n"
                        "Supporter: That sounds really tough. Have you been feeling this way for a long time?\n"
                        "Seeker: Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything. \n"
                        "The result is: Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything.: statement-emotion\n"
                        f"Now you need to annotate the dialogue behavior of the seeker's utterance {current_text} based on the conversation context {context}\n\n"
                        f"Important rules: Use only behavior from {behaviors}. If the utterance contains multiple dialogue behaviors, select the dialogue behavior that appears later.\n"
                        "The answer is:\n"}
        ]
    elif type == "stage_of_change":
        messages = [
            {"role": "system",
             "content": "You are an AI assistant that annotates stage of change in emotional support conversations. "
                        f"You must follow this rule: {stage_of_change_rules} to annotate the stage of change of the last seeker's utterance"
                        "Output format: The content of the last seeker's utterance: <stage of change>"
                        "Provide only the annotation without explanation."},
            {"role": "user",
             "content": f"You need to annotate the dialogue behavior following the example:"
                        "Seeker: Lately, I’ve been feeling so exhausted. No matter how much I sleep, I still wake up tired.\n"
                        "Supporter: That sounds really tough. Have you been feeling this way for a long time?\n"
                        "Seeker: Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything. \n"
                        "The result is: Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything.: contemplation\n"
                        f"Now you need to annotate the dialogue behavior of the seeker's utterance {current_text} based on the conversation context {context}\n\n"
                        f"Important rules: Use only stage from {stages}.\n"
                        "The answer is:\n"}
        ]
    print("introduce the fourth LLM")
    # print(f"prompt is{messages}")
    flag=True
    while flag:
        try:
            client = OpenAI(
                api_key="",
                base_url=""
            )
            completions = client.chat.completions.create(
                model="deepseek-r1",
                messages=messages,
                temperature=0,
            )
            flag = False
        except Exception as e:
            print("Some error happened here.")
            time.sleep(5)
    response = completions.choices[0].message.content.strip()
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = response.strip("\n+")
    response=response.strip()
    response=response.split(":")[-1]
    # 提取评分并返回
    if type=="trust":
        r,c,u=response.split(",")
        response=np.mean([int(r),int(c),int(u)])
    return response

def update_dataset_with_voting(annotated_dataset, raw_dataset,type):
    """
    更新原始数据集，基于三个标注数据集进行投票。
    返回更新后的数据集和需要第四个人标注的数据集。
    """
    updated_dataset = []
    need_fourth_annotation = []

    for raw_dialog in raw_dataset:
        updated_dialog = raw_dialog.copy()
        updated_turns = []

        for turn_index, turn in enumerate(raw_dialog["dialog"]):
            if turn["speaker"] == "seeker":
                # 提取三个数据集中对应轮次的情感强度
                intensity_list = []
                for dataset in annotated_dataset:
                    for dialog in dataset:
                        if dialog["index"] == raw_dialog["index"]:
                            for t in dialog["dialog"]:
                                if t["text"] == turn["text"] and t["speaker"] == "seeker":#原始数据集中的对话内容与标注的对话内容一致
                                    intensity_list.append(t[type])
                                    break
                            break

                # 进行投票
                final_intensity = vote_intensity(intensity_list)
                if final_intensity is not None:
                    turn[type] = final_intensity
                else:
                    # 需要第四个人标注
                    context = get_context(raw_dialog["dialog"], turn_index)
                    fourth_intensity = get_fourth_annotation(context, turn["text"],type)
                    turn[type] = fourth_intensity.strip()
                    need_fourth_annotation.append({
                        "index": raw_dialog["index"],
                        "text": turn["text"],
                        "speaker": "seeker",
                        "original_intensities": intensity_list,
                        "fourth_intensity": fourth_intensity,
                        "context": context
                    })

            updated_turns.append(turn)

        updated_dialog["dialog"] = updated_turns
        updated_dataset.append(updated_dialog)

    return updated_dataset, need_fourth_annotation

def integration_trust(raw_dataset):
    for raw_dialog in raw_dataset:
        for turn_index, turn in enumerate(raw_dialog["dialog"]):
            if turn["speaker"] == "seeker":
                if "reliability" in turn and "response_competence" in turn and "perceived_understandability" in turn:
                    turn["trust"]=np.mean([int(turn["reliability"]),int(turn["response_competence"]),int(turn["perceived_understandability"])])
    return raw_dataset
if __name__ == "__main__":

    original_data=json.load(open("./original_data/data_demo_10.json"))
    one = json.load(open("./annotated_data/stage_demo10_win6_one_llama.json"))
    two = json.load(open("./annotated_data/stage_demo10_win6_one_qwen.json"))
    three = json.load(open("./annotated_data/stage_demo10_win6_one_deepseek.json"))
    updated_dataset, need_fourth_annotation = update_dataset_with_voting([one,two,three], original_data,"stage_of_change")
    with open("./annotated_data/updated_dataset_demo10_stage.json", "w", encoding="utf-8") as f:
        json.dump(updated_dataset, f, ensure_ascii=False, indent=4)
    with open("./annotated_data/reliablity_need_fourth_annotation_stage.json", "w", encoding="utf-8") as f:
        json.dump(need_fourth_annotation, f, ensure_ascii=False, indent=4)
    #
    # 补充trust结果
    # integration_trust(original_data)
    # with open("./annotated_data/updated_dataset_emo_behavior_stage_re_co_per_trust.json", "w", encoding="utf-8") as f:
    #     json.dump(original_data, f, ensure_ascii=False, indent=4)
    # print("情感强度标注已完成，结果已保存至 updated_annotations.json")
