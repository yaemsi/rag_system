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
    output_dir: str = field(
        default="./output",
        metadata={
            "help": "Directory to save evaluation results."
        }
    )
    split: Literal["train", "train_a", "train_b", "train_c", "train_d", "valid", "bonus"] = field(
        default="valid",
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
    ret_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to evaluate the retrieval system or not."
        }
    )
    verbose: bool = field(
        default=False,
        metadata={
            "help": "Whether to print verbose evaluation output."
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
    embed_model: str = field(
        default="nomic-ai/nomic-embed-text-v1",
        metadata={
            "help": "Model ID served by the vLLM embedding server. "
                    "Must match the model name used when starting vLLM. "
                    "Examples: nomic-ai/nomic-embed-text-v1, "
                    "intfloat/multilingual-e5-large-instruct."
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
        default=20,
        metadata={
            "help": "Number of candidates returned by the retriever. "
                    "When reranking is enabled these are passed to the reranker "
                    "which then keeps only top_k_reader for the LLM. "
                    "Increase this (e.g. 20) to give the reranker more to work with."
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

    embed_base_url: str = field(
        default="http://localhost:8000/v1",
        metadata={
            "help": "Base URL of the vLLM embedding server "
                    "(OpenAI-compatible /v1/embeddings endpoint)."
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
class RerankerArguments:
    """
    Reranker parameters. The reranker sits between the retriever and the
    reader, re-scoring the top-k retrieved chunks with a cross-encoder for
    more accurate ranking before passing them to the LLM.
    """
    use_reranker: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply cross-encoder reranking after retrieval. "
                    "Disable to fall back to RRF-ranked retrieval order."
        }
    )
    rerank_model: str = field(
        default="Qwen/Qwen3-Reranker-4B",
        metadata={
            "help": "Model ID served by the vLLM reranker server. "
                    "Must match the model name used when starting vLLM. "
                    "Examples: Qwen/Qwen3-Reranker-4B, "
                    "cross-encoder/ms-marco-MiniLM-L-6-v2."
        }
    )
    top_k_reader: int = field(
        default=5,
        metadata={
            "help": "Number of chunks passed to the LLM reader after reranking. "
                    "Fewer, higher-quality chunks reduce noise in the context window. "
                    "When use_reranker=False, the first top_k_reader chunks from the "
                    "retriever are used instead."
        }
    )

    rerank_base_url: str = field(
        default="http://localhost:8002/v1",
        metadata={
            "help": "Base URL of the vLLM reranker server "
                    "(OpenAI-compatible /v1/completions endpoint)."
        }
    )

    def __post_init__(self) -> None:
        pass

@dataclass
class ReaderArguments:
    """
    Reader parameters.
    """
    generation_model: str = field(
        default="mistralai/Mistral-Nemo-Instruct-2407",
        metadata={
            "help": "Model ID served by the vLLM generation server. "
                    "Must match the model name used when starting vLLM. "
                    "Examples: mistralai/Mistral-Nemo-Instruct-2407, "
                    "Qwen/Qwen2.5-14B-Instruct."
        }
    )

    generation_base_url: str = field(
        default="http://localhost:8001/v1",
        metadata={
            "help": "Base URL of the vLLM generation server "
                    "(OpenAI-compatible /v1/chat/completions endpoint)."
        }
    )

    inference_delay: float = field(
        default=0.0,
        metadata={
            "help": "Seconds to sleep after each LLM call. "
                    "Set to 1.0–2.0 on sustained workloads to prevent "
                    "thermal crashes (e.g. RTX 5090 under full train eval)."
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