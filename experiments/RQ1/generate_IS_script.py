import pandas as pd
import os
import numpy as np
from huggingface_hub import login
from transformers import pipeline
from tqdm import tqdm

HF_token = os.environ.get("HF_TOKEN", "")

login(token=HF_token)
pipe = pipeline("text2text-generation", model="CNice/BART_is_generation_opt1")
df = pd.read_parquet("../../data/data_topic_and_target.parquet.gzip")
datasets = df["dataset"].unique()

for d in datasets:
    print("Dataset: ", d)
    cond = (df["dataset"] == d) & (df["implicitness"] == "yes")
    data = df[cond]
    gens = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        IS = pipe(row["text"])
        gens.append(IS)
    df.loc[cond, "IS"] = gens

df["IS_prep"] = df["IS"].apply(lambda t: t["generated_text"] if t is not np.nan else t)
df.to_parquet("../../data/data_IS_prep.parquet.gzip", index=False)
