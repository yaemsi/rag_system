# **Technical Challenge: Build a Question Answering System Over a Document Corpus**

## **Goal**

You will build a **question answering (QA) system** that, given:

* A corpus of documents, and  
* A set of queries.

Answers as many queries as possible with a **grounded answer** derived from the corpus.

This challenge is designed to evaluate your ability to:

* Explore and understand data.  
* Design a retrieval and answering strategy.  
* Build reliable evaluation metrics.  
* Produce clean, well-structured Python code.

We encourage a **Retrieval-Augmented Generation (RAG)** approach, but it is **not mandatory**. Creative solutions are welcome as long as answers are grounded in the corpus.

---

## **What You’ll Receive**

You will receive a zipped dataset containing:

### **1\) Documents**

A collection of documents. Each document has:

* A text field (main content).  
* A document id.  
* (Optional) Metadata fields

The documents are stored in a [corpus.jsonl.gz](http://corpus.jsonl.gz) file, in the format

```py
{
	id: str,
	text: str
	product_suffix: str | None
	product_version: str | None
	product_prefix: str | None
}
```

### **2\) Queries & Golden Evidence**

For all queries, we provide a valid/expected answer and the expected document id that answers the question. You will have access to the [train.jsonl.gz](http://train.jsonl.gz) and [valid.jsonl.gz](http://valid.jsonl.gz) files to develop your system. We keep an undisclosed test set to evaluate your solution against. The data is in the form

```py
{
	document_id: str,
	query: str,
	answer: str
}
```

**Optional ‘Bonus’ Queries**

Bonus / advanced questions are provided in a [bonus.jsonl.gz](http://bonus.jsonl.gz) file in the same format as the regular queries files. Bonus queries do not have a golden answer & document id. Some are answerable from the corpus, some are not \- in which case we expect the proposed Q & A system to answer accordingly (e.g. Sorry, I cannot answer that question).

---

## **Task Overview**

### **Part A — Exploration / EDA (warm-up)**

Start by familiarizing yourself with the dataset. Provide a short write-up (markdown or text) that includes:

* The dataset structure (documents, sizes)  
* Examples of documents and queries.  
* Anything notable (languages, noisy docs, metadata usefulness).  
* Quick stats (optional but appreciated).

---

### **Part B — Build the QA System**

Implement a system that takes a query and returns:

* **An answer** (string).  
* **The document ID that is supporting the answer.**

**Grounding requirement:** your answers must be based on the provided corpus, if a query cannot be answered from the corpus, your system must answer consequently.

---

## **Query Types You Should Expect**

All queries are answerable from the corpus, but may require:

* Decomposition  
* Aggregation  
* Multi-document reasoning

---

## **Constraints**

* **Language:** Your solution must be written in **Python**.  
* **LLM usage (optional):** If you use an LLM, you must use an **Ollama** model (we may provide an interface or you may assume a local Ollama runtime).  
* We provide a **code skeleton** including an evaluation loop.  
* You must implement **your own evaluation metric(s)** inside the skeleton. We will evaluate using **our own metrics** as well, so design choices should aim for robustness.

---

## **Deliverables**

Please submit:

1. **Code**  
* A runnable Python project (we prefer a simple structure: `src/`,  `pyproject.toml`).  
* Entry point script (e.g., `main.py`) that runs the evaluation loop from our skeleton.  
2. **Short write-up including:**  
* Your EDA analysis.  
* Your approach (retrieval \+ answering).  
* How you handle different query types.  
* Tradeoffs and assumptions.  
* What you would improve with more time.  
3. **Your evaluation metrics**  
* Implemented inside the provided skeleton.  
* Explain what they measure and why they’re relevant.

---

## **Evaluation Criteria**

We will assess:

### **1\) Correctness & grounding**

* Answers should be supported by the corpus.  
* Avoid hallucinations (especially if using an LLM).

### **2\) Robustness**

* Stable behavior across query types and noisy documents.  
* Reasonable defaults and fallback strategies.

### **3\) Engineering quality**

* Code clarity, modularity, reproducibility, and documentation.  
* Sensible dependencies and runtime considerations.

### **4\) Evaluation thoughtfulness**

* Useful metrics (even if imperfect).  
* Insight into failure modes (basic error analysis is a plus).

---

## **Solution Guidance**

Aim for a solution that is complete and clear rather than overly complex. Focus on correctness, grounding, and clean implementation.

Try to come up with a solution that answers as many queries as possible in the time you are willing to invest. It is likely you won’t have time to go through all of the requirements. In that case, come prepared to discuss with the team during the technical review. **It is very important that we receive a write-up (report) as described in the previous sections**. It will be our main source of information to prepare an insightful exchange with you during the technical challenge review. Be creative and have fun with the assignment!  
