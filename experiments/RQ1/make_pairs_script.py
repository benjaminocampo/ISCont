import pandas as pd
import os
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

datasets = all_data["dataset"].unique()
all_pairs = {}
threshold = 0.2

for d in datasets:
    df = all_data[all_data["dataset"] == d]
    imp_df = df[df["implicitness"] == "yes"]
    exp_df = df[df["implicitness"] == "no"]

    positive_pairs = []
    all_pairs[d] = []
    for row_id, imp_row in tqdm(imp_df.iterrows(), total=len(imp_df)):
        imp_text = imp_row["text"]
        imp_IS = imp_row["IS_prep"]
        imp_target = imp_row["target"]
        pairs = []
        exp_same_target = exp_df[exp_df['target'] == imp_target]
        for _, exp_row in exp_same_target.iterrows():
            exp_text = exp_row["text"]

            imp_IS_emb = imp_row["imp_IS_emb"]
            exp_emb = exp_row["text_emb"]

            cos_sim = euclidean_distances([imp_IS_emb], [exp_emb])
            if cos_sim < threshold:
                pairs[d].append((row_id, imp_text, exp_text, imp_IS))

        all_pairs[d].append(pairs)

for d in datasets:
    for xs in all_pairs[d]:
        all_data.iloc[xs[0], "pos_pairs"] = xs