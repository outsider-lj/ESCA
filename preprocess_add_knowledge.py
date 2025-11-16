import json
import faiss
import sqlite3
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re

encoder = SentenceTransformer("")


Knowledge_Mapping = {
    "information-related": ["self-disclosure", "information", "providing suggestions"],
    "emotion-related": ["affirmation and reassurance", "reflection of feelings"],
    "context-related": ["restatement or paraphrasing", "question", "others"]
}


def clean_text(text):
    """
    去除特殊符号和换行符，保留字母、数字、空格和常见标点符号。
    """

    cleaned_text= text.replace('\n', ' ').replace('\r', ' ')

    return cleaned_text


def get_strategy_category(strategy):
    for cat, strategies in Knowledge_Mapping.items():
        if strategy.lower() in strategies:
            return cat
    return None

#  SQLite + FAISS
def search_faiss_and_fetch_text(encoder, dial_history, index_path, db_path, top_k=1):
    # 加载索引
    index = faiss.read_index(index_path)

    # 编码查询
    if isinstance(dial_history,list):
        # user_query = " ".join([f"{t['speaker']}: {t['text']}" for t in dial_history])
        user_query = " ".join([f"{t['role']}: {t['content']}" for t in dial_history])
    else:
        user_query=dial_history
    query_vector = encoder.encode([user_query])
    query_vector = np.array(query_vector).astype("float32")

    # top_k
    distances, indices = index.search(query_vector, top_k)

    # SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    results = []
    for idx in indices[0]:
        cursor.execute("SELECT context,knowledge FROM knowledge WHERE faiss_index=?", (int(idx),))
        row = cursor.fetchone()
        if row:
            context=row[0]
            knowledge=row[1]
            results.append([context,knowledge])  # 只取 reframe 字段
    conn.close()

    return results

def query_chat_ai_model(model,messages,temperature=0.7):
    client = OpenAI(
        api_key="",
        base_url="",
    )

    completions = client.chat.completions.create(
        model=model,
        max_tokens=128,
        messages=messages,
        temperature=temperature,
    )
    output = completions.choices[0].message.content.strip()
    return output

# generate knowledge by LLM
def generate_context_knowledge(history,model):
    prompt = "Based on the given dialogue history, what does the current dialogue focus on? And what important information related to this topic not be metioned?" \
             "The answer is less than 200 words and without any format and analysis."
    if isinstance(history,str):
        usr_input = history
    else:
        user_input = " ".join([f"{t['speaker']}: {t['text']}" for t in history])
    messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}]
    knowledge=query_chat_ai_model(model,messages,0.7)
    return knowledge

def generate_emotion_knowledge(history,model):
    prompt = "Based on the given dialogue history, what is the current emotion problem the seeker focus on? And what emotianl reflection should the supporter provide to show the empathy based on the seeker's negative emotion and emotion problem?" \
             "The answer is less than 100 words and without any format and analysis."
    if isinstance(history,str):
        usr_input = history
    else:
        user_input = " ".join([f"{t['speaker']}: {t['text']}" for t in history])
    messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}]
    knowledge=query_chat_ai_model(model,messages,0.7)
    return knowledge


def enrich_dataset_with_knowledge_sqlite(
    input_path,
    output_path,
    reframing_index_path,
    reframing_db_path,
    psyqa_index_path,
    psyqa_db_path,
    emotion_index_path,
    emotion_db_path,
    chat_ai_model
):
    print("Loading dataset",flush=True)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for dialogue in tqdm(data):
        for i, turn in enumerate(dialogue["dialog"]):
            if turn["speaker"] == "seeker" :
                continue
            if turn["speaker"]=="supporter":
                strategy = dialogue["dialog"][i]["strategy"]
                category = get_strategy_category(strategy)
                if not category:
                    print(f"Some error in {strategy}")
                history = dialogue["dialog"][:i]
                # text = turn["text"]
                if category == "information-related":
                    # k1 = search_faiss_and_fetch_text(encoder, text, reframing_index_path, reframing_db_path)
                    k = search_faiss_and_fetch_text(encoder, history[-4:], psyqa_index_path, psyqa_db_path)
                    # k=k1+k2
                    k=[clean_text(s) for t in k for s in t]
                    turn["retrieved_knowledge"] = k

                elif category == "emotion-related":
                    k = search_faiss_and_fetch_text(encoder, history[-4:], emotion_index_path, emotion_db_path)
                    k=[clean_text(s) for t in k for s in t]
                    turn["retrieved_knowledge"] = k

                elif category == "context-related":
                    k = generate_context_knowledge(history,chat_ai_model)
                    turn["retrieved_knowledge"] = [clean_text(k)]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ====== Main 函数 ======
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process dialogue dataset with knowledge augmentation.")
    parser.add_argument("--input_path", type=str, default="", help="Path to input dialogue dataset (JSON)")
    parser.add_argument("--output_path", type=str,default="", help="Path to output file with added knowledge")

    parser.add_argument("--reframing_index", type=str, default="reframing_knowledge_faiss.index", help="Path to FAISS index for Reframing")
    parser.add_argument("--reframing_db_path", type=str,default="reframing_knowledge.db", help="CSV file for Reframing knowledge")

    parser.add_argument("--psyqa_index", type=str, default="psyqa_knowledge_faiss.index", help="Path to FAISS index for PsyQA")
    parser.add_argument("--psyqa_db_path", type=str,  default="psyqa_knowledge.db",help="CSV file for PsyQA knowledge")

    parser.add_argument("--emotion_index", type=str, default="emotional_knowledge_faiss.index", help="Path to FAISS index for Emotion-Reflection")
    parser.add_argument("--emotion_db_path", type=str,  default="emotional_knowledge.db", help="CSV file for Emotion-Reflection")
    parser.add_argument("--chat_ai_model", type=str, default="llama-3.3-70b-instruct")

    args = parser.parse_args()


    enrich_dataset_with_knowledge_sqlite(
        args.input_path, args.output_path,
        args.reframing_index, args.reframing_db_path,
        args.psyqa_index, args.psyqa_db_path,
        args.emotion_index, args.emotion_db_path,
        args.chat_ai_model
    )
