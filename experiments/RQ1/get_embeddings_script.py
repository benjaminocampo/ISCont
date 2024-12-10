import pandas as pd
import os
import backoff
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from mistralai import Mistral
from tqdm import tqdm

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

all_data = pd.read_parquet("../../data/data_IS_prep.parquet.gzip")
all_data["imp_IS_emb"] = all_data["IS_prep"].apply(lambda t: get_text_embedding(t) if t is not None else t)
all_data["text_emb"] = all_data["text"].apply(lambda t: get_text_embedding(t))
