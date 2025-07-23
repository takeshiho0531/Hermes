import argparse
import json
import time
import csv
import os
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_from_disk
from typing import Optional

def write_csv_row(output_file, fieldnames, row_dict):
    if not os.path.exists(output_file):
        with open(output_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row_dict)
    else:
        with open(output_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row_dict)


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

def perform_queries(index, retrieved_docs, embeddings, batch_size, max_batches=1000):
    query_times = []
    per_query_times = []
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
        batch_time = query_end - query_start
        query_times.append(batch_time)
        per_query_time = batch_time / len(batch)
        per_query_times.extend([per_query_time] * len(batch))

        retrieved_indices.append(I)

    retrieved_indices = np.vstack(retrieved_indices)  # shape = (num_queries, retrieved_docs)
    print(f"Retrieved indices shape: {retrieved_indices.shape}")

    return sum(query_times) / len(query_times) if query_times else 0, retrieved_indices, per_query_times

def run_faiss_retrieval_benchmark(
    index_name: str,
    nprobe_list: list[int],
    batch_size_list: list[int],
    retrieved_docs_list: list[int],
    num_threads_list: list[int],
    dataset_path: str,
    output_dir: str = "data/",
    query_embedding_path: Optional[str] = None,
    use_direct_embeddings: bool = False,
    embeddings: Optional[np.ndarray] = None,
) -> tuple[str, str]:

    if use_direct_embeddings:
        if embeddings is None:
            raise ValueError("when `use_direct_embeddings=True`, you need `embeddings`")
        query_base = "direct" 
    else:
        if query_embedding_path is None:
            raise ValueError("when `use_direct_embeddings=False`, you need `query_embedding_path`")
        query_base = os.path.splitext(os.path.basename(query_embedding_path))[0]
        embeddings = np.load(query_embedding_path)

    index_base = os.path.splitext(os.path.basename(index_name))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    profiling_dir = os.path.join(output_dir, "profiling")
    retrieved_dir = os.path.join(output_dir, "retrieved", f"{index_base}__{query_base}")
    os.makedirs(profiling_dir, exist_ok=True)
    os.makedirs(retrieved_dir, exist_ok=True)

    output_file = os.path.join(
        profiling_dir, 
        f"retrieval_monolithic_latency__{index_base}__{query_base}__{timestamp}.csv"
    )

    # Initially load the index with a dummy nprobe; it will be updated later in the loop.
    index = load_faiss_index(index_name, nprobe_list[0])

    print(f"loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"dataset loaded, number of documents: {len(dataset)}")
    
    with open(output_file, mode='w', newline='') as file:
        fieldnames = ["Index Name", "nprobe", "Batch Size", "Retrieved Docs", "Num Threads", "Avg Retrieval Time (s)"]
        timer_writer = csv.DictWriter(file, fieldnames=fieldnames)
        timer_writer.writeheader()
        
        for nprobe in tqdm(nprobe_list, desc="nprobe values", position=0):
            index.nprobe = nprobe  # Update the index's nprobe
            for batch_size in tqdm(batch_size_list, desc=f"Batch sizes (nprobe={nprobe})", position=1, leave=False):
                for retrieved_docs in tqdm(retrieved_docs_list, desc=f"Retrieved Docs (nprobe={nprobe}, batch_size={batch_size})", position=2, leave=False):
                    for num_threads in tqdm(num_threads_list, desc=f"Num Threads (nprobe={nprobe}, batch_size={batch_size}, retrieved_docs={retrieved_docs})", position=3, leave=False):
                        # Set the FAISS thread count
                        faiss.omp_set_num_threads(num_threads)
                        # Measure the average query time for the current combination
                        avg_query_time, retrieved_indices, per_query_times = perform_queries(
                            index, retrieved_docs, embeddings, batch_size,
                        )
                        setting_tag = f"nprobe{nprobe}_bs{batch_size}_k{retrieved_docs}_nt{num_threads}_{timestamp}"
                        setting_retrieved_dir = os.path.join(retrieved_dir, setting_tag)
                        os.makedirs(setting_retrieved_dir, exist_ok=True)
                        np.save(os.path.join(setting_retrieved_dir, f"retrieved_doc_indices.npy"), retrieved_indices)
                        retrieved_texts_dict = {}
                        for i, doc_ids in enumerate(retrieved_indices):
                            texts = [dataset[int(doc_id)]["raw"] for doc_id in doc_ids]
                            retrieved_texts_dict[str(i + 1)] = texts
                        with open(os.path.join(setting_retrieved_dir, f"retrieved_texts.json"), "w") as f:
                            json.dump(retrieved_texts_dict, f, indent=2)

                        write_csv_row(
                            output_file,
                            fieldnames,
                            {
                                "Index Name": index_name,
                                "nprobe": nprobe,
                                "Batch Size": batch_size,
                                "Retrieved Docs": retrieved_docs,
                                "Num Threads": num_threads,
                                "Avg Retrieval Time (s)": avg_query_time
                            }
                        )


                        query_time_path = os.path.join(setting_retrieved_dir, "per_query_times.csv")
                        with open(query_time_path, 'w', newline='') as f_time:
                            timer_writer = csv.writer(f_time)
                            timer_writer.writerow(["Query ID", "Query Time (s)"])
                            for i, qt in enumerate(per_query_times):
                                timer_writer.writerow([i+1, qt])


    print(f"✅ Results saved to {profiling_dir, retrieved_dir}")
    return output_file, retrieved_dir



def main():
    args = parse_arguments()

    run_faiss_retrieval_benchmark(
        index_name=args.index_name,
        nprobe_list=args.nprobe,
        batch_size_list=args.batch_size,
        retrieved_docs_list=args.retrieved_docs,
        num_threads_list=args.num_threads,
        query_embedding_path=args.queries,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
