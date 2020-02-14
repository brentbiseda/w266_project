# W266 Final Project - Natural Language Processing 

## Brent Biseda and Katie Mo 
## Spring 2020 

# Installation Instructions  

## CUDA Install  

cuda_10.0.130_411.31_win10.exe

## Pip install  

pip install --upgrade pip
pip install -r requirements.txt



# Bio Bert

## Start up Tensorboard and set directory

tensorboard --logdir ../output/biobert_finetuned

## Go to Tensorboard location  

http://localhost:6006

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=10 --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.1_pubmed/model.ckpt-1000000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_finetuned

## Run Prediction for Sentiment Analysis on Development Set  

python run_classifier.py --do_eval=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.1_pubmed/model.ckpt-1000000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_dev

## Run Prediction for Sentiment Analysis on TestSet  

python run_classifier.py --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.1_pubmed/model.ckpt-1000000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_test

# Bert Cased

## Start up Tensorboard and set directory

tensorboard --logdir ../output/bert_cased_finetuned

## Go to Tensorboard location  

http://localhost:6006

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=10 --vocab_file=../model/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/cased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/cased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/bert_cased_finetuned

