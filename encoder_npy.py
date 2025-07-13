from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import faiss


dataset = load_dataset("cais/mmlu", name="abstract_algebra", split="test")

texts = dataset["question"]

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") 

embeddings = model.encode(texts, batch_size=32, show_progress_bar=True) # needs to be 768 dim

save_path = "queries/mmlu_abstract_algebra_dev.npy"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.save(save_path, embeddings)

print(f"✅ Saved {len(embeddings)} embeddings to: {save_path}")

# dim check
index = faiss.read_index("data/indices/monolithic_indices/hermes_index_monolithic_sphere_10k_exported.faiss")
queries = np.load("queries/mmlu_abstract_algebra_dev.npy")

print("Index dimension:", index.d)
print("Query embedding shape:", queries.shape)
