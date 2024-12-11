import pandas as pd
import os
import backoff
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from mistralai import Mistral
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

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


IS_embs = []
text_embs = []
for _, row in tqdm(all_data.iterrows(), total=len(all_data)):    
    if row["IS_prep"] is not None:
        IS_emb = get_text_embedding(row["IS_prep"])
        IS_embs.append(IS_emb)
    else:
        IS_embs.append(None)
    
    text_emb = get_text_embedding(row["text"])
    text_embs.append(text_emb)

all_data["IS_emb"] = IS_embs
all_data["text_emb"] = text_embs

all_data.to_parquet("../../data/data_IS_embs.parquet.gzip", index=False)