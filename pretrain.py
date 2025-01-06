import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "False"
import random
import sys
sys.path.append("./")
import numpy as np
import torch

from llm.method import modelPretrain

import json  
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import load_dataset, concatenate_datasets

from omegaconf import OmegaConf

# DONE: reproduce RoBERTa numbers on the Longformer corpus
# DONE: testing ddp single machine
# DONE: testing ddp multiple machines
# DONE: testing resume from checkpoint
# TODO: try on a TPU-pod
# TODO: run on beaker on ai2-server1/2





    
def add_args(parser):
    
    # HF model loading
    #parser.add_argument("--tokenizer", type=str, default='roberta-base')
    #parser.add_argument("--model", type=str, default='roberta-base')

    # Checkpointing and logging
    parser.add_argument("--save_dir", type=str, default='data/training_data/')
    parser.add_argument("--save_prefix", type=str, default='train')
    #parser.add_argument("--resume", type=str, default=None)
    #parser.add_argument("--resume_model_only", type=str, default=None)
    #parser.add_argument("--log_rate", type=int, default=10)
    #parser.add_argument("--disable_checkpointing", type=bool, default=False)
    # Training hyperparams
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--num_training_steps", type=int, default=200)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    #parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--logging_steps", type=int, default=1)
    # datasets
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--data_type", type=str, required=True)
    #parser.add_argument("--train_dev_split", type=float, default=0.05)
    # model settings
    parser.add_argument("--freeze_llm", type=bool, default=False)
    parser.add_argument("--freeze_mlp", type=bool, default=False)
    parser.add_argument("--config_path", required=True)
    #
    args = parser.parse_args()
    conf = OmegaConf.load(args.config_path)
    extra_conf = OmegaConf.create(vars(args))
    conf = OmegaConf.merge(conf, extra_conf)
    if not hasattr(conf.model, "tokenizer_path"):
        conf.model.tokenizer_path = conf.model.path
    if not hasattr(conf.merge, "path"):
        conf.merge.path =None
    return conf


def main(args):
    """
    random.seed(args.seed * 10)
    np.random.seed(args.seed * 100)
    torch.manual_seed(args.seed * 1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed * 10000)
    
    
    """
    output_dir = args.save_dir+args.save_prefix+'_'+str(args.merge.length)+'_'+str(args.merge.n_layer)
    tokenizer=AutoTokenizer.from_pretrained(args.model.tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    


    # Load dataset
    def tokenize_function(examples):
        inputs = []
        for input_text, context, answer in zip(examples['context'], examples['input'], examples['answers']):
            inputs.append({
                    "role": "user",
                    "content": "question:"+input_text+"context:"+context
                })
            inputs.append({
                    "role": "system",
                    "content": "answer:"+answer
                })
        model_inputs = tokenizer(inputs, truncation=True)
        return model_inputs
                
        """
        inputs = [f"question:{input_text}context:{context}answer:{answer}" for input_text, context, answer 
                                                        in zip(examples['context'], 
                                                            examples['input'], 
                                                            examples['answers'])]
        model_inputs = tokenizer(inputs, truncation=True)
        return model_inputs
        
        """

         
    def tokenize_function_2(examples):
        inputs = [json.dumps(message) for message in examples['messages']]
        return tokenizer(inputs, truncation=True)
    
    def tokenize_function_1(examples):
        return tokenizer(examples['messages'], truncation=True)
    
    def filter_function(example):
        return len(example['text']) > 2048
    
    """
    
    
    """
    dataset = load_dataset(args.datasets, split="train")
    tokenized_datasets = dataset.map(tokenize_function_2, batched=True)

    """
    ,
                    
                    
                    
                    "narrativeqa","passage_retrieval_en","passage_retrieval_en_e",
                    
    # 加载多个数据集
                    
    dataset_names = ["2wikimqa","2wikimqa_e","hotpotqa","hotpotqa_e","multifieldqa_en","multifieldqa_en_e",
                    "musique","qasper","qasper_e","qmsum"]


    datasets = [load_dataset('THUDM/LongBench', name, split='test') for name in dataset_names]
    dataset = concatenate_datasets(datasets)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    
    """
    
    

    # 合并数据集
    


    filtered_train_dataset = tokenized_datasets#.filter(filter_function)
    train_dataset = filtered_train_dataset.shuffle(seed=42).select(range(1200))
    eval_dataset = filtered_train_dataset.shuffle(seed=42).select(range(1))

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    pass
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs
    )
    
    # Initialize Trainer
    print("Initialize Trainer")
    pretrainer = modelPretrain(
        args=args,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer

    )
    
    """
    optimizer = pretrainer.optimizer
    schedular = pretrainer.lr_scheduler
        # 查看优化器参数
    print(optimizer)
    for param_group in optimizer.param_groups:
        print(param_group)
        if hasattr(param_group, 'shape'):
            print(param_group.shape)
    print(schedular)
    
    """
    
    # Train model
    try:
        print("Train model")
        pretrainer.train()
    finally:
        pretrainer.save_model(output_dir)
    
    
    
    

    
if __name__ == "__main__":
    args = add_args(argparse.ArgumentParser(description="pretrain"))
    main(args = args)
