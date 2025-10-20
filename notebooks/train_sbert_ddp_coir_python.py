import os
import pathlib
import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
from collections import defaultdict
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    SentenceTransformer,
    InputExample,
)
from transformers import AutoTokenizer

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# === DDP setup ===
local_rank = int(os.environ.get("LOCAL_RANK", 0))
def setup_ddp():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    print(f"DDP initialized: Running on GPU {local_rank} of {torch.cuda.device_count()}", flush=True)
setup_ddp()
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

# === Only train on Python ===
language = "python"
print(f"Training only on language: {language}")

# === Load dataset for Python ONLY ===
def load_dataset_once(dataset_name, split):
    # Same logic as your distributed download function
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if rank == 0:
        print(f"[Rank {rank}] Downloading dataset...")
        dataset = load_dataset(dataset_name, split=split)
        if world_size > 1:
            dist.barrier()
    else:
        print(f"[Rank {rank}] Waiting for dataset to be downloaded...")
        dist.barrier()
        dataset = load_dataset(dataset_name, split=split)

    return dataset

queries = load_dataset_once(f"CoIR-Retrieval/CodeSearchNet-ccr-{language}-queries-corpus", split="queries")
corpus = load_dataset_once(f"CoIR-Retrieval/CodeSearchNet-ccr-{language}-queries-corpus", split="corpus")
qrels = load_dataset_once(f"CoIR-Retrieval/CodeSearchNet-ccr-{language}-qrels", split="train")

# === No prefix needed since only one language ===
corpus_dict = {str(row["_id"]): row["text"] for row in corpus}
qrels_dict = defaultdict(list)
for item in qrels:
    qrels_dict[str(item["query_id"])].append(str(item["corpus_id"]))

train_examples = []
for item in queries:
    query_id = str(item["_id"])
    query_text = item["text"]
    for positive_id in qrels_dict.get(query_id, []):
        if positive_id in corpus_dict:
            positive_text = corpus_dict[positive_id]
            train_examples.append(InputExample(texts=[query_text, positive_text]))

print("Number of training examples:", len(train_examples))

# === Convert to HuggingFace Dataset ===
train_data = [{"query": ex.texts[0], "text": ex.texts[1]} for ex in train_examples]
train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})

# === Model setup ===
model_name = "ibm-granite/granite-embedding-125m-english"
model = SentenceTransformer(model_name)

save_dir = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "output",
    f"{model_name}-CodeSearchNetCCRetrieval-{language}-lr1e-5-epochs10-temperature20_full_dev"
)
os.makedirs(save_dir, exist_ok=True)

# === Wrap with DDP if needed ===
if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

training_args = SentenceTransformerTrainingArguments(
    output_dir=save_dir,
    num_train_epochs=10,
    #max_steps=100, 
    per_device_train_batch_size=16,
    learning_rate=1e-5,
    warmup_steps=int(len(train_dataset) * 0.1 / 16),
    #warmup_steps = 0,
    logging_steps=1,
    #save_steps=100,
    save_strategy="epoch",
    evaluation_strategy="no",
    save_total_limit=10,
    dataloader_num_workers=8,
    ddp_find_unused_parameters=True
)

trainer = SentenceTransformerTrainer(
    model=model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
    args=training_args,
    train_dataset=train_dataset,
    loss=losses.MultipleNegativesRankingLoss(model=model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model),
    eval_dataset=None,
    callbacks=[]
)

print(f"[Rank {dist.get_rank()}] Starting training with {len(train_dataset)} examples")

trainer.train()

dist.destroy_process_group()

