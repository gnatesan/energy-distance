import os
import pathlib
import torch
import torch.distributed as dist
from datasets import load_dataset, concatenate_datasets, Dataset
from collections import defaultdict
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses, SentenceTransformer, InputExample
from transformers import AutoTokenizer

#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

# ** Get local rank **
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# ** Initialize DDP **
def setup_ddp():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    print(f"DDP initialized: Running on GPU {local_rank} of {torch.cuda.device_count()}")

#setup_ddp()

# ** Set device **
#device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

def load_dataset_once(dataset_name, split):
    # Initialize distributed mode if not already initialized
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Only rank 0 downloads the dataset
    if rank == 0:
        print(f"[Rank {rank}] Downloading dataset...")
        dataset = load_dataset(dataset_name, split=split)
        if world_size > 1:
            dist.barrier()  # Notify others that download is complete
    else:
        print(f"[Rank {rank}] Waiting for dataset to be downloaded...")
        dist.barrier()  # Wait for rank 0 to finish downloading
        dataset = load_dataset(dataset_name, split=split)

    return dataset


languages = ["go", "java", "python", "javascript", "php", "ruby"]
#languages = ["python"]


all_queries, all_corpus, all_qrels = [], [], []

for lang in languages:
    queries = load_dataset(f"CoIR-Retrieval/CodeSearchNet-ccr-{lang}-queries-corpus", split="queries")
    corpus = load_dataset(f"CoIR-Retrieval/CodeSearchNet-ccr-{lang}-queries-corpus", split="corpus")
    qrels = load_dataset(f"CoIR-Retrieval/CodeSearchNet-ccr-{lang}-qrels", split="train")

    # Prefix IDs with language name to avoid collisions
    queries = queries.map(lambda x: {"_id": f"{lang}_{x['_id']}"})
    corpus = corpus.map(lambda x: {"_id": f"{lang}_{x['_id']}"})
    qrels = qrels.map(lambda x: {
        "query_id": f"{lang}_{x['query_id']}",
        "corpus_id": f"{lang}_{x['corpus_id']}"
    })
    
    all_queries.append(queries)
    all_corpus.append(corpus)
    all_qrels.append(qrels)

merged_queries = concatenate_datasets(all_queries)
merged_corpus = concatenate_datasets(all_corpus)
merged_qrels = concatenate_datasets(all_qrels)

print(f"Sample from merged_corpus: {merged_corpus[0]}")
print("Number of training queries:", len(merged_queries))
print("Size of corpus", len(merged_corpus))

# Load model
#model_name = "gte-modernbert-base"  # Change if needed
model_name = "ibm-granite/granite-embedding-125m-english"
model = SentenceTransformer("ibm-granite/granite-embedding-125m-english")
#model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")
#tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-embedding-125m-english")
#model.max_seq_length = 512

MAX_LENGTH = 512

# Step 1: Index corpus and qrels
corpus_dict = {doc["_id"]: doc["text"] for doc in merged_corpus}

# Group all relevant corpus ids by query id
qrels_dict = defaultdict(list)
for item in merged_qrels:
    qrels_dict[item["query_id"]].append(item["corpus_id"])

# Step 2: Create InputExample list
train_examples = []

for item in merged_queries:
    query_id = item["_id"]
    query_text = item["text"]

    # Add a training example for each positive corpus ID
    for positive_id in qrels_dict.get(query_id, []):
        if positive_id in corpus_dict:
            positive_text = corpus_dict[positive_id]
            train_examples.append(InputExample(texts=[query_text, positive_text]))

# Check the structure of the dataset
#print(raw_dataset["train"].features)  # Shows the features (columns) of the dataset
#print(raw_dataset["train"][0])  # Shows the first example in the 'train' dataset


print("Number of training examples: ", len(train_examples))


train_data = [{"query": example.texts[0], "text": example.texts[1]} for example in train_examples]
train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})


#print(f"[Rank {dist.get_rank()}] Dataset size before sharding: {len(train_dataset)}")

#train_dataset = train_dataset.shard(num_shards=dist.get_world_size(), index=dist.get_rank())

#print(f"[Rank {dist.get_rank()}] Dataset size after sharding: {len(train_dataset)}")


#def tokenize(example):
#    return tokenizer(
#        example["query"],
#        example["text"],
#        truncation=True,
#        max_length=MAX_LENGTH,
#        padding="max_length"
#    )

#train_dataset = train_dataset.map(tokenize, batched=True)


# ** Set save_dir **
save_dir = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "output",
    f"{model_name}-CodeSearchNetCCRetrieval_ED-lr2e-5-epochs10-temperature10_full_dev"
)
os.makedirs(save_dir, exist_ok=True)

# ** Wrap model in DDP if needed **
#if torch.cuda.device_count() > 1:
#    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

# ** Define training arguments **
training_args = SentenceTransformerTrainingArguments(
    output_dir=save_dir,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    #max_steps = 100,
    #warmup_steps = 10,
    warmup_steps=int(len(train_dataset) * 0.1 / 16),
    logging_steps=1,
    save_strategy="epoch",           # Save after each epoch
    evaluation_strategy="no",        # No evaluation
    save_total_limit=10,
    dataloader_num_workers=8,
    ddp_find_unused_parameters=True
)

# ** Train **
trainer = SentenceTransformerTrainer(
    model=model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model,
    args=training_args,
    train_dataset=train_dataset,
    loss=losses.MultipleNegativesRankingLoss(model=model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model),
    eval_dataset=None,
    callbacks=[]
)

# Check if we're in a distributed environment
if torch.distributed.is_available() and torch.distributed.is_initialized():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f"[Rank {rank}/{world_size}] Starting training with {len(train_dataset)} examples")
else:
    print(f"Starting single-process training with {len(train_dataset)} examples")


#print(f"[Rank {dist.get_rank()}] Starting training with {len(train_dataset)} examples")

trainer.train()

# ** Cleanup DDP **
#dist.destroy_process_group()

