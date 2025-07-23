import argparse
import os
import numpy as np  # type: ignore
from datetime import datetime
from datasets import load_dataset, Dataset  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from typing import Union, Optional, List, Literal
from transformers import AutoTokenizer, AutoModel  # type: ignore
import torch  # type: ignore


class BaseEmbedder:
    def encode(
        self,
        texts: list[str] | str,
        convert_to_tensor: bool = True,
        **kwargs,
    ):
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: list[str] | str,
        convert_to_tensor: bool = True,
        **kwargs,
    ):
        return self.model.encode(
            texts, convert_to_tensor=convert_to_tensor, **kwargs
        )


class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "google-bert/bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(
        self,
        texts: list[str] | str,
        convert_to_tensor: bool = True,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = output.last_hidden_state.mean(
            dim=1
        )  # (batch_size, hidden_size)

        return (
            embeddings if convert_to_tensor else embeddings.cpu().numpy()
        )


def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Query Benchmark")
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset to encode"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="Model type for encoding questions",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name for encoding questions",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="queries/",
        help="Directory to save encoded queries",
    )
    parser.add_argument(
        "--rewriter-used",
        type=bool,
        default=False,
        help="Whether to use a rewriter for the questions",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to use"
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        default=None,
        help="Subset name for the dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--questions",
        type=str,
        nargs="*",
        default=None,
        help="List of questions to encode",
    )
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
        questions = list(dataset["question"])
        

    if model_type == "sentence_transformer":
        model = SentenceTransformer(model_name)

    elif model_type == "huggingface":
        model = HuggingFaceEmbedder(model_name)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Use 'sentence_transformer' or 'huggingface'."
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

    save_name = f"{subset_str}_{model_tag}{rewriter_tag}_{timestamp}.npy"
    os.makedirs(save_subdir, exist_ok=True)
    save_path = os.path.join(save_subdir, save_name)
    np.save(save_path, embeddings)

    print(f"Saved {len(embeddings)} embeddings to: {save_path}")
    return save_path


def encode_question(
    encoding_model: BaseEmbedder,
    question: str,
    save_dir: str = "queries",
    save: bool = True,
    rewriter_used: bool = True,
    batch_size: int = 32,
) -> Union[str, np.ndarray]:

    embeddings = encoding_model.encode(
        question, batch_size=batch_size, show_progress_bar=True
    )
    if embeddings.ndim == 1:
        embeddings = embeddings[np.newaxis, :]  # shape: (1, 768)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        filename = "rewriter.npy" if rewriter_used else "no_rewriter.npy"
        path = os.path.join(save_dir, filename)

        np.save(path, embeddings)
        print(f"Saved: {path}")
        return path

    return embeddings


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
