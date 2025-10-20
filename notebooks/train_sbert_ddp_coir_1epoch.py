import os
import pathlib
import glob
import torch
import torch.distributed as dist
from datasets import load_dataset, concatenate_datasets, Dataset
from collections import defaultdict
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    SentenceTransformer,
    InputExample,
)

# ------------------------
# Helpers
# ------------------------
def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return not is_dist() or dist.get_rank() == 0

# LOCAL_RANK provided by torchrun
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# ------------------------
# (Optional) initialize CUDA device (HF/Trainer will init DDP internally)
# ------------------------
# If you prefer to force device pinning up-front:
torch.cuda.set_device(local_rank)

# ------------------------
# Load CodeSearchNet-CCR exactly like before
# ------------------------
languages = ["go", "java", "python", "javascript", "php", "ruby"]
all_queries, all_corpus, all_qrels = [], [], []

for lang in languages:
    queries = load_dataset(f"CoIR-Retrieval/CodeSearchNet-ccr-{lang}-queries-corpus", split="queries")
    corpus  = load_dataset(f"CoIR-Retrieval/CodeSearchNet-ccr-{lang}-queries-corpus", split="corpus")
    qrels   = load_dataset(f"CoIR-Retrieval/CodeSearchNet-ccr-{lang}-qrels",           split="train")

    # Prefix IDs with language to avoid collisions
    queries = queries.map(lambda x, l=lang: {"_id": f"{l}_{x['_id']}"})
    corpus  = corpus.map( lambda x, l=lang: {"_id": f"{l}_{x['_id']}"})
    qrels   = qrels.map(  lambda x, l=lang: {"query_id": f"{l}_{x['query_id']}",
                                             "corpus_id": f"{l}_{x['corpus_id']}"})
    all_queries.append(queries)
    all_corpus.append(corpus)
    all_qrels.append(qrels)

merged_queries = concatenate_datasets(all_queries)
merged_corpus  = concatenate_datasets(all_corpus)
merged_qrels   = concatenate_datasets(all_qrels)

if is_main_process():
    print(f"Sample from merged_corpus: {merged_corpus[0]}")
    print("Number of training queries:", len(merged_queries))
    print("Size of corpus", len(merged_corpus))

# ------------------------
# Build pair list (query, positive) exactly as before
# ------------------------
model_name = "ibm-granite/granite-embedding-125m-english"
model = SentenceTransformer(model_name)
MAX_LENGTH = 512

corpus_dict = {doc["_id"]: doc["text"] for doc in merged_corpus}

qrels_dict = defaultdict(list)
for item in merged_qrels:
    qrels_dict[item["query_id"]].append(item["corpus_id"])

train_examples = []
for item in merged_queries:
    qid = item["_id"]
    qtext = item["text"]
    for pos_id in qrels_dict.get(qid, []):
        if pos_id in corpus_dict:
            train_examples.append(InputExample(texts=[qtext, corpus_dict[pos_id]]))

if is_main_process():
    print("Number of training examples:", len(train_examples))

train_data = [{"query": ex.texts[0], "text": ex.texts[1]} for ex in train_examples]
train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})

# ------------------------
# Save directory (same structure as your runs)
# NOTE: model_name contains a '/', so this creates .../output/ibm-granite/granite-embedding-125m-english-...
# ------------------------
save_dir = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "output",
    f"{model_name}-CodeSearchNetCCRetrieval_ED-lr2e-5-epochs10-temperature10_full_dev",
)
os.makedirs(save_dir, exist_ok=True)

# ------------------------
# Training args: 10 epochs total, save each epoch (so resuming at 9 → 1 epoch left)
# Since training for 10 epochs showed continuous increases in validation performance after each epoch, target is now 15 epochs
# ------------------------
training_args = SentenceTransformerTrainingArguments(
    output_dir=save_dir,
    num_train_epochs=15,                   # total target
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=int(len(train_dataset) * 0.1 / 16) if len(train_dataset) > 0 else 10,
    logging_steps=1,
    save_strategy="epoch",
    evaluation_strategy="no",
    save_total_limit=20,
    dataloader_num_workers=8,
    ddp_find_unused_parameters=True,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    loss=losses.MultipleNegativesRankingLoss(model=model),
    eval_dataset=None,
    callbacks=[],
)

# ------------------------
# Resume from the 9-epoch checkpoint and finish epoch 10
#Now resume from 10-epoch and finish 15 epochs
# ------------------------
# Default checkpoint path you gave:
explicit_resume = os.path.join(
    save_dir,
    "checkpoint-70950"  # after 10 epochs
)

resume_path = os.environ.get("RESUME_FROM", explicit_resume)

if is_main_process():
    print(f"Requested resume checkpoint: {resume_path}")

if os.path.isdir(resume_path):
    if is_main_process():
        print(f"Resuming training from: {resume_path}")
    trainer.train(resume_from_checkpoint=resume_path)
else:
    # Fallback: if directory naming differs, allow auto-detect of latest
    if is_main_process():
        print("Specified checkpoint not found; trying to auto-resume from latest in output_dir (if any).")
    # True = HF Trainer picks latest checkpoint under output_dir
    trainer.train(resume_from_checkpoint=True)

# ------------------------
# Save a final copy explicitly as 'checkpoint-70950'
# (ensures you have that exact folder even if HF’s auto step naming differs)
# ------------------------
#final_tag = "checkpoint-70950"
#final_path = os.path.join(save_dir, final_tag)

#if is_main_process():
#    try:
#        # Prefer trainer.save_model so the HF artifacts are written properly
#        trainer.save_model(final_path)
#        print(f"✅ Saved final model to: {final_path}")
#    except Exception as e:
#        print(f"Warning: trainer.save_model failed with {e}; attempting SentenceTransformer.save()")
#        try:
#            model.save(final_path)
#            print(f"✅ Saved SentenceTransformer model to: {final_path}")
#        except Exception as ee:
#            print(f"❌ Could not save final model copy: {ee}")

# If distributed, let ranks sync before exit to avoid early teardown
if is_dist():
    dist.barrier()

