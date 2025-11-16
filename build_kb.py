import faiss
import numpy as np
import json
import sqlite3
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import argparse
# convert csv data to database and related index

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def build_faiss_index(encoder,knowledge_name,data, index_path="knowledge_faiss.index"):
    if "reframing" in knowledge_name:
        texts = data['situation'].tolist()
    elif "emotion" in knowledge_name:
        texts = data['seeker_post'].tolist()
    elif "psy" in knowledge_name:
        texts = data['description'].tolist()
    embeddings = encoder.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    faiss.write_index(index, index_path)

    return index

import json
import re

def clean_text(text):

    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)  # 合并多个空格
    text = text.strip().lower()

    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s,.?!:;\'\"“”‘’()\[\]{}\-]', '', text)
    return text


# 构建 SQL 数据库
def build_sql_database(knowledge_name,data, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge(
                        id INTEGER PRIMARY KEY, 
                        faiss_index INTEGER,
                        context TEXT, 
                        knowledge TEXT
                        )''')
    if "reframing" in knowledge_name:
        for idx, row in data.iterrows():
            try:
                cursor.execute(
                    "INSERT INTO knowledge (faiss_index, context,knowledge) VALUES (?, ?, ?)",
                    (idx, row['thought'], clean_text(row['reframe']))
                )
                print(f"插入第{idx}数据：{row['thought']}")
            except Exception as e:
                print(f"插入失败: {e}")
    elif "emotion" in knowledge_name:
        for idx, row in data.iterrows():
            try:
                cursor.execute(
                    "INSERT INTO knowledge (faiss_index,context,knowledge) VALUES (?, ?, ?)",
                    (idx, row['seeker_post'], clean_text(row['response_post']))
                )
                print(f"插入第{idx}数据：{row['seeker_post']}")
            except Exception as e:
                print(f"插入失败: {e}")
    elif "psy" in knowledge_name:
        for idx, row in data.iterrows():
            try:
                cursor.execute(
                    "INSERT INTO knowledge (faiss_index,context,knowledge) VALUES (?, ?, ?)",
                    (idx, row['description'], clean_text(row['answers']))
                )
                print(f"insert the {idx} item：{row['description']}")
            except Exception as e:
                print(f"insert fail: {e}")
    conn.commit()
    conn.close()

def check_columns(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM knowledge WHERE faiss_index=1")
    columns = [row for row in cursor.fetchall()]

    conn.close()

    print("数据库表字段:", columns)
    return "faiss_index" in columns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--knowledge_name', type=str, default="emotion")
    parser.add_argument('--knowledge_path', type=str, default="./emotional-reactions-reddit.csv")#emotional_reactions-reddit#translated_part_0#new_reframing_dataset
    parser.add_argument('--vector_path', type=str, default="./emotional_knowledge.db")
    parser.add_argument('--index_path', type=str, default="emotional_knowledge_faiss.index")
    # add_model_args(parser)
    args = parser.parse_args()
    # introduce Sentence Transformer model
    encoder = SentenceTransformer(args.model_path)
    data = load_data(args.knowledge_path)
    #build SQL database
    build_sql_database(args.knowledge_name,data, db_path=args.vector_path)
    #build FAISS index
    faiss_index = build_faiss_index(encoder,args.knowledge_name,data, index_path=args.index_path)

    if check_columns(args.vector_path):
        print(" `faiss_index` 字段存在！")
    else:
        print("`faiss_index` 字段缺失！")
# daic_woz_data = load_daic_woz_data()
# faiss_index, faiss_texts = build_faiss_index(cognitive_data)
# build_sql_database(cognitive_data, "knowledge.db")