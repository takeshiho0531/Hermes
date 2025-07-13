import argparse
import json
import time
import csv
import os
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_from_disk

def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Query Benchmark")
    parser.add_argument("--index-name", type=str, required=True, help="Path to the FAISS index file")
    parser.add_argument("--nprobe", type=int, nargs='+', required=True, help="List of nprobe values for FAISS search")
    parser.add_argument("--batch-size", type=int, nargs='+', required=True, help="List of batch sizes for querying")
    parser.add_argument("--queries", type=str, required=True, help="Path to the NumPy file containing embeddings")
    parser.add_argument("--retrieved-docs", type=int, nargs='+', required=True, help="List of numbers of docs retrieved per query")
    parser.add_argument("--num-threads", type=int, nargs='+', required=True, help="List of numbers of threads to run retrieval")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the HF dataset directory (load_from_disk)")
    parser.add_argument("--output-dir", type=str, default="data/", help="Directory where the results will be saved")
    return parser.parse_args()

def load_faiss_index(index_name, nprobe):
    index = faiss.read_index(index_name)
    index.nprobe = nprobe
    return index

def perform_queries(index, retrieved_docs, embeddings, batch_size, retrieved_dir, max_batches=1000):
    query_times = []
    retrieved_indices = []
    
    # Progress bar for the query batches (position=4)
    for idx in tqdm(range(0, min(len(embeddings), max_batches * batch_size), batch_size),
                    desc="Querying batches",
                    leave=False,
                    position=4):
        batch = embeddings[idx:idx + batch_size]
        query_start = time.time()
        _, I = index.search(batch, retrieved_docs)
        query_end = time.time()
        query_times.append(query_end - query_start)
        retrieved_indices.append(I)

    retrieved_indices = np.vstack(retrieved_indices)  # shape = (num_queries, retrieved_docs)
    print(f"Retrieved indices shape: {retrieved_indices.shape}")

    os.makedirs(retrieved_dir, exist_ok=True)
    np.save(os.path.join(retrieved_dir, "retrieved_doc_indices.npy"), retrieved_indices)

    return sum(query_times) / len(query_times) if query_times else 0, retrieved_indices

def main():
    args = parse_arguments()

    index_base = os.path.splitext(os.path.basename(args.index_name))[0]
    query_base = os.path.splitext(os.path.basename(args.queries))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    profiling_dir = os.path.join(args.output_dir, "profiling")
    retrieved_dir = os.path.join(args.output_dir, "retrieved", f"{index_base}__{query_base}__{timestamp}")
    os.makedirs(profiling_dir, exist_ok=True)
    os.makedirs(retrieved_dir, exist_ok=True)

    output_file = os.path.join(
        profiling_dir, 
        f"retrieval_monolithic_latency__{index_base}__{query_base}__{timestamp}.csv"
    )
    
    # Initially load the index with a dummy nprobe; it will be updated later in the loop.
    index = load_faiss_index(args.index_name, args.nprobe[0])
    embeddings = np.load(args.queries)
    
    with open(output_file, mode='w', newline='') as file:
        fieldnames = ["Index Name", "nprobe", "Batch Size", "Retrieved Docs", "Num Threads", "Avg Retrieval Time (s)"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for nprobe in tqdm(args.nprobe, desc="nprobe values", position=0):
            index.nprobe = nprobe  # Update the index's nprobe
            for batch_size in tqdm(args.batch_size, desc=f"Batch sizes (nprobe={nprobe})", position=1, leave=False):
                for retrieved_docs in tqdm(args.retrieved_docs, desc=f"Retrieved Docs (nprobe={nprobe}, batch_size={batch_size})", position=2, leave=False):
                    for num_threads in tqdm(args.num_threads, desc=f"Num Threads (nprobe={nprobe}, batch_size={batch_size}, retrieved_docs={retrieved_docs})", position=3, leave=False):
                        # Set the FAISS thread count
                        faiss.omp_set_num_threads(num_threads)
                        # Measure the average query time for the current combination
                        avg_query_time, retrieved_indices = perform_queries(
                            index, retrieved_docs, embeddings, batch_size, retrieved_dir
                        )
                        writer.writerow({
                            "Index Name": args.index_name,
                            "nprobe": nprobe,
                            "Batch Size": batch_size,
                            "Retrieved Docs": retrieved_docs,
                            "Num Threads": num_threads,
                            "Avg Retrieval Time (s)": avg_query_time
                        })
                        file.flush()  # Ensure data is written incrementally

    dataset = load_from_disk(args.dataset_path)

    all_retrieved_texts = []  # shape: (num_queries, top_k)
    for doc_ids in retrieved_indices:
        texts = [dataset[int(doc_id)]["raw"] for doc_id in doc_ids]
        all_retrieved_texts.append(texts)

    with open(os.path.join(retrieved_dir, "retrieved_texts.json"), "w") as f:
        json.dump(all_retrieved_texts, f, indent=2)  

    print(f"✅ Results saved to {profiling_dir, retrieved_dir}")

if __name__ == "__main__":
    main()
