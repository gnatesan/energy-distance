# Energy Distance for RAG Systems in LLMs and Sentence Transformers

## Project Description
This project explores the application of [energy distance](https://en.wikipedia.org/wiki/Energy_distance) as an alternative metric to [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) in RAG systems used with LLMs and Sentence Transformers. The aim is to improve the accuracy and relevance of retrieved documents by using a metric that potentially captures finer nuances in sentence embeddings.

Our approach capitalizes on existing benchmarking frameworks, traditionally used to evaluate the efficacy of cosine similarity with fully trained LLMs. We have introduced custom implementations of energy distance into these well-established, open-source frameworks. By employing the same benchmarks universally recognized in the research community, specifically Massive Text Embedding Benchmark ([MTEB](https://huggingface.co/blog/mteb#:~:text=MTEB%20is%20a%20massive%20benchmark,on%20a%20variety%20of%20tasks.)), we establish a robust, scientific, and quantifiable basis to assess whether energy distance offers any substantive advantages over cosine similarity. Benchmarks currently utilized originate from [Hugging Face](https://huggingface.co) and [SBERT](https://www.sbert.net). Our modifications to these frameworks are accessible via the following links:

**[Sentence Transformer with Energy Distance](https://github.com/gnatesan/sentence-transformers-energydistance)**

We have extended the functionality of the Sentence Transformer by integrating an energy distance calculation into the Contrastive Loss module. This adaptation aims to minimize the energy distance between embeddings of similar sentences during model training, promoting more effective similarity measures. Additionally, to enhance query representation, we adapted the model to generate multivector embeddings for queries, ensuring detailed and nuanced semantic capture.

**[MTEB Evaluator](https://github.com/gnatesan/mteb-evaluator)**

This includes enhancements to the Massive Text Embedding Benchmark, a tool designed to assess information retrieval systems. We incorporated energy distance as a novel metric to refine the selection of top matches for a given query within this framework. To accommodate multivector queries against single-vector documents, we modified the query encoding process, generating token embeddings that better capture the complexity of the queries.

## Repository Outline
- `README.md`: Description of repository contents
- `requirements.txt`: Required python packages
- `/models`: Trained sentence transformer models
- `/notebooks`: Source code used to train sentence transformers and to run benchmarks
- `/results`: Benchmark results

## How to Run

**Pre-requisites:**
- GPU VRAM >= 12GB
- Python >= 3.8

```bash
pip install -r requirements.txt
```

To run the code, follow the instructions included in the python notebooks under `/notebooks`. Benchmarks can be run from MTEB.ipynb. Sentence transformer training can be run from train_energy_sentence_transformer.ipynb.


## Results (NEEDS TO BE UPDATED, NOT VALID)

We focused on retrieval Recall. This is a measure of a model's ability to find all the relevant cases within a dataset.
* recall_at_1000 calculates how many relevant documents are found in the top 1000


#| Model | MTEB Benchmark | Cosine Similarity | Energy Distance (Untrained Sentence Transformer) | Energy Distance (Trained Sentence Transformer)
|----------|:--------:|---------:|---------:|---------:|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)     | ArguAna : recall_at_1000   | 0.99502     |  0.99573  |  0.95448  |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)     | ArguAna : recall_at_1000  | 0.99573     |  0.99573  |  0.98933  |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)     | ArguAna : recall_at_1000   | 0.99644     |  0.99431  |  0.99431  |
| [multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)     | ArguAna : recall_at_1000   | 0.99502     |  0.99502  |  0.97653  |
| [nq-distilbert-base-v1](https://huggingface.co/sentence-transformers/nq-distilbert-base-v1)     | ArguAna : recall_at_1000   | 0.98293     | 0.97653  |  0.97653  |


* evaluation_time is in seconds. Note that Cosine Similarity sometimes used benchmarks that were precalculated, giving them unrealistically fast evaluation times. 

| Model | MTEB Benchmark | Cosine Similarity | Energy Distance (Untrained Sentence Transformer) | Energy Distance (Trained Sentence Transformer)
|----------|:--------:|---------:|---------:|---------:|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)     | ArguAna : evaluation_time   | 8.59     |  1562.93  |  1110.49  |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)     | ArguAna : evaluation_time   | 9.51     |  1103.01  |  1115.91  |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)     | ArguAna : evaluation_time   | 53.88     |  1997.01  |  1645.77  |
| [multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)     | ArguAna : evaluation_time   | 12.49     |  1172.29  |  1107.27  |
| [nq-distilbert-base-v1](https://huggingface.co/sentence-transformers/nq-distilbert-base-v1)     | ArguAna : evaluation_time   | 27.58     | 1488.82  |  1125.27  |
