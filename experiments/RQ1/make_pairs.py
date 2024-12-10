# %%
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from mistralai import Mistral
import os
import backoff  # for exponential backoff

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

client = Mistral(api_key=api_key)

@backoff.on_exception(backoff.expo, Exception)
def get_text_embedding(inputs):
    embeddings_batch_response = client.embeddings.create(
        model=model,
        inputs=inputs
    )
    return embeddings_batch_response.data[0].embedding
# %%
import pandas as pd
from tqdm import tqdm

all_data = pd.read_parquet("../../data/data_IS_prep.parquet.gzip")

datasets = all_data["dataset"].unique()
# %%
all_data["imp_IS_emb"] = all_data["IS_prep"].apply(lambda t: get_text_embedding(t) if t is not None else t)
# %%
all_data["text_emb"] = all_data["text"].apply(lambda t: get_text_embedding(t))
# %%
all_pairs = {}
for d in datasets:
    df = all_data[all_data["dataset"] == d]
    imp_df = df[df["implicitness"] == "yes"]
    exp_df = df[df["implicitness"] == "no"]

    threshold = 0.2
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

            imp_IS_emb = get_text_embedding(imp_IS)
            exp_emb = get_text_embedding(exp_text)

            #import pdb; pdb.set_trace()
            cos_sim = euclidean_distances([imp_IS_emb], [exp_emb])
            if cos_sim < threshold:
                pairs[d].append((row_id, imp_text, exp_text, imp_IS))

        all_pairs[d].append(pairs)


    #all_data.loc[
    #    (all_data["implicitness"] == "yes") & (all_data["dataset"] == d),
    #    "positive_pairs",
    #] = positive_pairs
# df.loc[df["implicitness"] == "no", "negative_pairs"] = negative_pairs
# %%
all_data
# %%
