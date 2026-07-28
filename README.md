# Advanced PDF Retrieval & RAG Optimization

A notebook-based project that turns a raw mortgage PDF into a searchable, question-answering system using Retrieval-Augmented Generation (RAG).

<!-- TODO: add a screenshot of a sample query + grounded answer here -->

## What It Does

Give it a PDF - a mortgage Lender Fees Worksheet, in this case - and it extracts the text page by page, indexes it with LlamaIndex, and answers questions like "what is the total estimated monthly payment" with grounded answers instead of guesses.
It experiments with different ways of finding the right passage: vector search, BM25 keyword search, and a hybrid of the two, then reranks the candidates with a cross-encoder before handing them to Gemini to generate the final answer.
Query expansion rephrases the question a few different ways first, so retrieval isn't limited to the user's exact wording.

## Tech Stack

- Python
- Jupyter Notebook
- LlamaIndex
- Google Gemini API
- HuggingFace Sentence Transformers
- PyMuPDF
- Pandas / Matplotlib

## Install and Run

```bash
git clone <url> RAG
cd RAG
python -m venv venv
source venv/bin/activate
pip install llama-index google-generativeai sentence-transformers pymupdf pandas matplotlib
jupyter notebook

Open the notebook, select the venv kernel, and run the cells top to bottom.
```
What I Learned

- Designing a RAG pipeline instead of just calling an LLM - separated the pipeline into ingestion, indexing, retrieval, reranking, and generation, and saw how each layer affects answer quality.
- Working with LlamaIndex abstractions - explored how Documents, VectorStoreIndex, retrievers, and post-processors fit together, and how to swap components without rewriting everything.
- Semantic vs. keyword vs. hybrid retrieval - compared vector search, BM25, and a hybrid approach, and saw concrete trade-offs on legal/financial text.
- Using rerankers to improve answer grounding - a cross-encoder reranker meaningfully improves relevance by re-scoring candidate chunks before they reach the LLM.
