# %%
from huggingface_hub import login
import os

HF_token = os.environ.get("HF_TOKEN", "")
HF_token
# %%
login(token=HF_token)
# %%
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text2text-generation", model="CNice/BART_is_generation_opt1")
# %%
import pandas as pd
df = pd.read_parquet("../../data/data_topic_and_target.parquet.gzip")
# %%
datasets = df["dataset"].unique()
# %%
from tqdm import tqdm

for d in datasets:
    data = df[df["dataset"] == d]
    gens = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        IS = pipe(row["text"])
        gens.append(IS)
    df.loc[df["dataset"] == d, "IS"] = gens
# %%
df.to_parquet("../../data/data_IS.parquet.gzip", index=False)