from huggingface_hub import login
import pandas as pd
import os
from transformers import pipeline
from tqdm import tqdm

HF_token = os.environ.get("HF_TOKEN", "")
HF_token

login(token=HF_token)
pipe = pipeline("text2text-generation", model="CNice/BART_is_generation_opt1")
df = pd.read_parquet("../../data/data_topic_and_target.parquet.gzip")
datasets = df["dataset"].unique()

for d in datasets:
    data = df[df["dataset"] == d].sample(5)
    gens = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        IS = pipe(row["text"])
        gens.append(IS)
    df.loc[df["dataset"] == d, "IS"] = gens

df.to_parquet("../../data/data_IS.parquet.gzip", index=False)
