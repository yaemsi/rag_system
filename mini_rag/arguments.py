from dataclasses import dataclass, field
from typing import Literal
 
 
@dataclass
class GeneralArguments:
    """
    Whether to preprocess data, train or evaluate
    """
    data_dir: str = field(
        default="./data/gz",
        metadata={
            "help": "Directory containing the data files (corpus.jsonl.gz, train.jsonl.gz, valid.jsonl.gz, bonus.jsonl.gz)."
        }
    )
    index_dir: str = field(
        default="./data/index",
        metadata={
            "help": "Directory containing the index files."
        }
    )
    split: Literal["train", "valid", "bonus"] = field(
        default="train",
        metadata={
            "help": "Which split to evaluate on."
        }
    )
    rebuild: bool = field(
        default=False,
        metadata={
            "help": "Whether to force re-indexing of the corpus."
        }
    )
    eval_batch_size: int = field(
        default=16,
        metadata={
            "help": "Batch size for evaluation."
        }
    )
    top_k_eval: int = field(
        default=10,
        metadata={
            "help": "Number of retrieved passages to provide to the reader during evaluation."
        }
    )
    output_dir: str = field(
        default="./output",
        metadata={
            "help": "Directory where result JSON files are saved. "
                    "File name will be results-{split}.json."
        }
    )
    ret_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to evaluate the retrieval system or not."
        }
    )
    verbose: bool = field(
        default=False,
        metadata={
            "help": "Whether to print per-question token_f1 scores after QA evaluation."
        }
    )
    def __post_init__(self):
        pass
 
@dataclass
class ChunkerArguments:
    """
    Chunker parameters.
    """
    chunk_size: int = field(
        default=800,
        metadata={
            "help": "Size of each chunk."
        }
    )
    chunk_overlap: int = field(
        default=150,
        metadata={
            "help": "Overlap between consecutive chunks."
        }
    )
    min_split_len: int = field(
        default=850,
        metadata={
            "help": "Minimum document length to consider splitting."
        }
    )
    def __post_init__(self):
        pass
 
 
@dataclass
class RetrieverArguments:
    """
    Retriever parameters.
    """
    embed_model: Literal["nomic-embed-text", "mxbai-embed-large"] = field(
        default="nomic-embed-text",
        metadata={
            "help": "Ollama embedding model to use."
        }
    )
    embed_dim: int = field(
        default=768,
        metadata={
            "help": "Dimension of the embedding model output. "
                    "nomic-embed-text=768, mxbai-embed-large=1024."
        }
    )
    top_k_dense: int = field(
        default=20,
        metadata={
            "help": "Number of candidates from dense search before RRF."
        }
    )
    top_k_sparse: int = field(
        default=20,
        metadata={
            "help": "Number of candidates from BM25 before RRF."
        }
    )
    top_k_final: int = field(
        default=10,
        metadata={
            "help": "Number of final documents returned to the reader."
        }
    )
    rrf_k: int = field(
        default=60,
        metadata={
            "help": "RRF smoothing constant."
        }
    )
    embedding_batch_size: int = field(
        default=64,
        metadata={
            "help": "Batch size for embedding procedure."
        }
    )
 
    def __post_init__(self):
        # Auto-set embed_dim if model changed but dim was left at default
        model_dims = {
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
        }
        if self.embed_model in model_dims:
            self.embed_dim = model_dims[self.embed_model]
 
 
 
@dataclass
class ReaderArguments:
    """
    Reader parameters.
    """
    generation_model: Literal["mistral-nemo", "qwen2.5:7b", "qwen2.5:14b"] = field(
        default="mistral-nemo",
        metadata={
            "help": "Reader architecture."
        }
    )
 
    low_confidence_threshold: float = field(
        default=0.005,
        metadata={
            "help": "Threshold for low-confidence retrievals."
        }
    )
 
    max_tokens: int | None = field(
        default=None,  #4096 for qwen2.5:7b, 8192 for mistral-nemo and qwen2.5:14b
        metadata={
            "help": "Maximum tokens for generation."
        }
    )
    num_ctx: int | None = field(
        default=None,  #4096 16384 32768 65536
        metadata={
            "help": "Context window size for generation."
        }
    )
    temperature: float = field(
        default=0.0, # deterministic for grounded QA
        metadata={
            "help": "ampling temperature (0.0 = deterministic)."
        }
    )
 
    def __post_init__(self):
        pass
