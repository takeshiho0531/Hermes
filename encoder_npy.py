import argparse
import os
import numpy as np
from datetime import datetime
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from typing import Union, Optional, List, Literal

def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Query Benchmark")
    parser.add_argument("--dataset", type=str, required=True, help="dataset to encode")
    parser.add_argument("--model-type", type=str, required=True, help="Model type for encoding questions")
    parser.add_argument("--model-name", type=str, required=True, help="Model name for encoding questions")
    parser.add_argument("--save-dir", type=str, default="queries/", help="Directory to save encoded queries")
    parser.add_argument("--rewriter-used", type=bool, default=False, help="Whether to use a rewriter for the questions")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--subset-name", type=str, default=None, help="Subset name for the dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--questions", type=str, nargs='*', default=None, help="List of questions to encode")
    return parser.parse_args()

def encode_dataset_questions(
    dataset: Union[str, Dataset],
    model_name: str,
    model_type: Literal["sentence_transformer", "huggingface"],
    save_dir: str = "queries",
    rewriter_used: bool = False,
    split: str = "test",
    subset_name: Optional[str] = None,
    batch_size: int = 32,
    questions: Optional[List[str]] = None,
) -> str:
    if isinstance(dataset, str):
        dataset = load_dataset(dataset, name=subset_name, split=split)

    if questions is None:
        questions = dataset["question"]

    if model_type == "sentence_transformer":
        model = SentenceTransformer(model_name)

    elif model_type == "huggingface":
        raise NotImplementedError(
            "model_type='huggingface' is not yet supported in this function."
        )

    embeddings = model.encode(
        questions, batch_size=batch_size, show_progress_bar=True
    )

    subdir = "mmlu"  # TODO
    save_subdir = os.path.join(save_dir, subdir)

    # Filename formatting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subset_str = subset_name or "unknown"
    rewriter_tag = "_rewritten" if rewriter_used else ""
    model_tag = model_name.split("/")[-1]

    save_name = f"{subset_str}__{model_tag}{rewriter_tag}__{timestamp}.npy"
    os.makedirs(save_subdir, exist_ok=True)
    save_path = os.path.join(save_subdir, save_name)
    np.save(save_path, embeddings)

    print(f"Saved {len(embeddings)} embeddings to: {save_path}")
    return save_path

def main():
    args = parse_arguments()

    encode_dataset_questions(
        dataset=args.dataset,
        model_name=args.model_name,
        model_type=args.model_type,
        save_dir=args.save_dir,
        rewriter_used=args.rewriter_used,
        split=args.split,
        subset_name=args.subset_name,
        batch_size=args.batch_size,
        questions=args.questions,
    )

if __name__ == "__main__":
    main()