# Project Overview
This research project explores the use of energy distance as a retrieval metric in dense information retrieval systems, with a focus on improving Retrieval-Augmented Generation (RAG) workflows. Unlike the traditional single-vector based cosine similarity, energy distance captures the full statistical distribution of multivector query embeddings, offering a more expressive similarity measure. Our work builds on earlier experiments conducted with the BEIR benchmark (specifically HotpotQA dataset) where energy distance demonstrated competitive performance with cosine similarity. More recently, we developed an efficient implementation of energy distance using torch.einsum, enabling parallel computation of multivector queries against single-vector document embeddings with minimal GPU overhead. To handle scalability challenges, we introduced dynamic query batching and padding-based tensor alignment, making it feasible to run full-batch evaluations across large corpora. These optimizations allow for multi-hop and complex queries to be better represented in embedding space. Encouraged by strong results on HotpotQA, especially with longer and more information-rich queries, we are now shifting our focus to the CodeSearchNet-CCR dataset under the COIR benchmark, which contains similarly long-form and structured queries. Preliminary results suggest that energy distance continues to outperform cosine similarity in these scenarios, supporting the hypothesis that multivector representations enhance retrieval quality in real-world QA and code search tasks. 

To support this work, we extended two major libraries:

sentence-transformers-3.4.1 was modified to support token-level query embeddings during encoding and evaluation, as well as to integrate custom distance metrics like energy distance within the training and inference pipelines.
**https://github.com/gnatesan/sentence-transformers-3.4.1**

mteb-1.34.14 was customized to support energy distance scoring and batched multivector evaluations across BEIR-format datasets, ensuring compatibility with standard IR benchmarks while enabling direct comparisons against cosine-based baselines. **https://github.com/gnatesan/mteb-1.34.14**


# Energy Distance Project Training and Inference Instructions

## Setting up Python Environment and Installing Required Libraries
1. conda create --name myenv39 python=3.9
2. conda activate myenv39
3. pip install --upgrade pip --index-url https://pypi.org/simple
4. pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
5. git clone https://github.com/gnatesan/sentence-transformers-3.4.1.git
6. git clone https://github.com/gnatesan/mteb-1.34.14.git
7. pip install -e /path_to_sentence-transformers/sentence-transformers-3.4.1
8. pip install -e /path_to_mteb/mteb-1.34.14
9. git clone https://github.com/gnatesan/beir.git

## Sanity Check
1. conda create --name testenv python=3.9
2. conda activate testenv
3. pip install --upgrade pip --index-url https://pypi.org/simple
4. pip install sentence-transformers==3.4.1
5. pip install mteb==1.34.26
6. sbatch inference_CosSim.sh (Make sure the batch script calls eval_dataset.py and a baseline model is being used. *i.e. model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")*)
7. Cross reference the inference results with what is on the leaderboard. https://huggingface.co/spaces/mteb/leaderboard

## Model Training
1. cd /path_to_beir/beir/examples/retrieval/training
2. Before running training, make sure the model, model_name, and hyperparameters (LR, scale) are correct. 
nano train_sbert_latest_2.py or nano train_sbert_ddp_2.py to change model, model_name, and LR. 
nano sentence-transformers-3.4.1/sentence-transformers/losses/MultipleNegativesRankingLoss.py to change scale. 
3. sbatch train.sh OR sbatch train_ddp.sh if using multiple GPUs
4. Trained model will be saved in /path_to_beir/beir/examples/retrieval/training/output

## Model Evaluation
1. sbatch inference_ED.sh if evaluating an ED trained model (myenv39 conda environment must be setup)
2. sbatch inference_CosSim.sh if evaluating a cosine similarity trained model (testenv conda environment must be setup)
3. Make sure the proper python script in the batch file is being run (if evaluating entire dataset or subset based on query lengths)

## RPI Cluster Setup
1. ssh <username>@blp01.ccni.rpi.edu
2. ssh nplfen01 (Access NPL front-end node x86)
3. cd ~/barn
4. Follow instructions to install Conda on x86 https://docs.cci.rpi.edu/software/Conda/ (Make sure conda is installed in barn directory)
5. echo 'export PATH="$HOME/miniconda3x86/condabin:$PATH"' >> ~/.bashrc
6. source ~/.bashrc 
7. export http_proxy=http://proxy:8888
export https_proxy=$http_proxy\
source /gpfs/u/home/MSSV/MSSVntsn/barn/miniconda3x86/etc/profile.d/conda.sh\
export TMPDIR=~/barn\
export TRANSFORMERS_CACHE=/gpfs/u/home/MSSV/MSSVntsn/barn\
export HF_HOME=/gpfs/u/home/MSSV/MSSVntsn/barn\
(Step 7 needs to be done every time you log in to the node, replace MSSVntsn with your username)


## IMPORTANT FILES
1. train.sh - Batch script to run model training on a single GPU.  
2. train_ddp.sh - Batch script to run model training on multiple GPUs. Make sure number of GPUs requested are properly set.
3. inference_ED.sh - Batch script to run inference on an ED trained model. Can run on either entire dataset or subset based on query lengths.
4. inference_CosSim.sh Batch script to run inference on a CosSim trained model. Can run on either entire dataset or subset based on query lengths.
5. train_sbert_latest_2.py - Python script to run model training on a single GPU. Uses ir_evaluator to evaluate on a dev set after each epoch of training and only saves the best model, make sure ir_evaluator is enabled.
6. train_sbert_ddp_2.py - Python script to run model training on multiple GPUs using PyTorch DDP. Currently does not use an ir_evaluator to evaluate on a dev set after each epoch of training.
7. train_sbert_ddp_coir.py - Python script to run model training for CodeSearchNet-CCR from COIR benchmark. Correctly preprocesses the 6 different languages in the dataset (go, python, java, javascript, ruby, php). Runs on multiple GPUs, no ir_evaluator used for dev set evaluation.
8. train_sbert_ddp_coir_1epoch.py - Python script to load a saved model checkpoint and run model training for CodeSearchNet-CCR from COIR benchmark. Correctly preprocesses the 6 different languages in the dataset (go, python, java, javascript, ruby, php). Runs on multiple GPUs, no ir_evaluator used for dev set evaluation.
9. train_sbert_ddp_coir_python.py - Python script run model training for only one programming language in CodeSearchNet-CCR from COIR benchmark. Language used for training is python but that can be changed. Runs on multiple GPUs, no ir_evaluator used for dev set evaluation.
10. eval_dataset.py - Python script to run inference on entire BEIR dataset.
11. eval_dataset_subset_length.py - Python script to run inference on subset of BEIR dataset based on query lengths.

## IMPORTANT NOTES
1. All files used for training should be present when you clone the gnatesan/beir repository in beir/examples/retrieval/training folder.
2. If working on the RPI cluster NPL node make sure that all installations occur in the ~/barn directory due to larger memory storage. 
