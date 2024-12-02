# %%
# Use a pipeline as a high-level helper
import pandas as pd
from transformers import pipeline

pipe = pipeline("text2text-generation", model="CNice/BART_is_generation_opt1")
# %%
import pandas as pd
df = pd.read_parquet("../../data/data_IS.parquet.gzip")
# %%
imp_df = df[df["implicitness"] == "yes"]
exp_df = df[df["implicitness"] == "no"]
# %%
threshold = 1e-3
all_pairs = []
for _, imp_row in imp_df.iterrows():
    imp_text = imp_row["text"]
    imp_IS = imp_row["IS"]
    # For each implicit message, pair it with and explicit one

    pairs = [] # List[Tuple[Str]]
    for _, exp_row in exp_df.iterrows():
        exp_text = exp_row["text"]

        imp_IS_emb = emb(imp_IS)
        exp_IS_emb = emb(exp_text)
        
        cos_sim = similarity(imp_IS, exp_text)
        if cos_sim < threshold:
            pairs.append(exp_text)
    
    all_pairs.append(pairs)

df.loc[df["implicitness"] == "yes", "pairs"] = all_pairs