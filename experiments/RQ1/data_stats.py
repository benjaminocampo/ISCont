# %%
import pandas as pd

df = pd.read_parquet("../../data/data_topic_and_target.parquet.gzip")
# %%
df
# %%
df.columns
# %%
df["dataset"].value_counts()
# %%
print(pd.crosstab(df["dataset"], df["implicitness"]).to_latex())
# %%
