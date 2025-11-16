import os
import json
import sys
from metric import NLGEval
# from metric.myMetrics import Metric
import pandas as pd
import nltk
# from nltk.util import ngrams
from nltk import word_tokenize
def calc_distinct_n(n, candidates, print_score: bool = True):
    dict = {}
    total = 0
    # candidates = [word_tokenize(candidate) for candidate in candidates]
    candidates_lst=[]
    for candidate in candidates:
        try:
            candidates_lst.append(word_tokenize(candidate))
        except:
            print(candidate)
    for sentence in candidates_lst:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i : i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    return score
eval_file='all_results.csv'

df = pd.read_csv(eval_file)  # 替换为你的实际文件路径
labels=list(df.columns.values)
num_columns=len(labels)

metric = NLGEval()
for i in range(7,num_columns):
    label=labels[i]
    hyps = df[label].tolist()
    refs = df['gold_response'].tolist()
    hyp_list=[]
    ref_list = []
    for ref,hyp in zip(refs,hyps):
        if pd.isna(hyp):
            continue
        try:
            hyp_list.append(' '.join(nltk.word_tokenize(hyp.lower())))
            ref_list.append(' '.join(nltk.word_tokenize(ref.lower())))
        except:
            print(hyp)

    print("Begin calculate")
    dist1=calc_distinct_n(1,hyps)
    dist2=calc_distinct_n(2,hyps)
    dist3=calc_distinct_n(3,hyps)
    Dist={"dist-1":dist1,
          "dist-2": dist2,
          "dist-3": dist3,
          }
    print(f"Distinct of {label}")
    print(Dist)
    metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list)
    metric_res.update(Dist)
    with open(os.path.join("evaluation", f'{label}_metric_nlgeval.json'), 'w') as f:
        json.dump(metric_res, f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join("evaluation", f'{label}_metric_nlgeval_list.json'), 'w') as f:
        json.dump(metric_res_list, f, ensure_ascii=False)