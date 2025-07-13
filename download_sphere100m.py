from datasets import load_dataset, Dataset
import os

# 設定
STREAM_LIMIT = 100_000_000 
CHUNK_SIZE = 100_000
SAVE_ROOT = "/share7/akiho.kawada/sphere_chunks_100k"
CACHE_DIR = "/share7/akiho.kawada/sphere_100m/"

os.makedirs(SAVE_ROOT, exist_ok=True)

print("📥 Streaming from mohdumar/SPHERE_100M ...")
streamed = load_dataset(
    "mohdumar/SPHERE_100M",
    split="train",
    streaming=True,
    cache_dir=CACHE_DIR
)

examples = []
current_chunk = 0
start_index = 0

for i, item in enumerate(streamed):
    if i >= STREAM_LIMIT:
        break

    if i % 10000 == 0:
        print(f"⏳ Processed {i} samples...")

    examples.append(item)

    if len(examples) == CHUNK_SIZE:
        chunk_path = os.path.join(SAVE_ROOT, f"chunk_{current_chunk:03d}")
        if os.path.exists(chunk_path):
            print(f"⚠️ Skipping existing {chunk_path}")
        else:
            print(f"💾 Saving chunk {current_chunk} to {chunk_path} ...")
            subset = Dataset.from_list(examples)
            subset.save_to_disk(chunk_path)
            print(f"✅ Chunk {current_chunk} saved.")
        current_chunk += 1
        examples = []

if examples:
    chunk_path = os.path.join(SAVE_ROOT, f"chunk_{current_chunk:03d}")
    print(f"💾 Saving final chunk {current_chunk} to {chunk_path} ...")
    subset = Dataset.from_list(examples)
    subset.save_to_disk(chunk_path)
    print(f"✅ Final chunk {current_chunk} saved.")
