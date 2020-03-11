# W266 Final Project - Natural Language Processing 

## Brent Biseda and Katie Mo 
## Spring 2020 

# Installation Instructions  

## CUDA Install  

cuda_10.0.130_411.31_win10.exe

## Pip install  

pip install --upgrade pip
pip install -r requirements.txt

# Start up Tensorboard and set directory

tensorboard --logdir ../output

## Go to Tensorboard location  

http://localhost:6006

# Bio Bert 1.1

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=2 --do_lower_case=False --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.1_pubmed/model.ckpt-1000000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.1_finetuned

## Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_1.1_finetuned/model.ckpt-20162 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.1_finetuned

# Bio Bert 1.0

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=2 --do_lower_case=False --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.0_pubmed_pmc/biobert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.0_finetuned

## Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_1.0_finetuned/model.ckpt-20162 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.0_finetuned

# Bert Cased

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=2 --vocab_file=../model/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/cased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/cased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/bert_cased_finetuned

## Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_cased_finetuned/model.ckpt-20162 --data_dir=../datasets/SENT/ --output_dir=../output/bert_cased_finetuned

# Bert Un-Cased

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=2 --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=True --init_checkpoint=../model/uncased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/bert_uncased_finetuned

## Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_uncased_finetuned/model.ckpt-20162 --data_dir=../datasets/SENT/ --output_dir=../output/bert_uncased_finetuned

# Clinical Bert - Biobert_pretrain_output_disch_100000

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=2 --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_disch_100000_finetuned

## Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_pretrain_output_disch_100000_finetuned/model.ckpt-20162 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_disch_100000_finetuned

# Clinical Bert - Biobert_pretrain_output_all_notes_150000

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=2 --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_all_notes_150000_finetuned

## Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_pretrain_output_all_notes_150000_finetuned/model.ckpt-20162 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_all_notes_150000_finetuned

# Clinical Bert - bert_pretrain_output_disch_100000

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=2 --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_disch_100000_finetuned

## Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_pretrain_output_disch_100000_finetuned/model.ckpt-20162 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_disch_100000_finetuned

# Clinical Bert - bert_pretrain_output_all_notes_150000

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=2 --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_all_notes_150000_finetuned

## Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_pretrain_output_all_notes_150000_finetuned/model.ckpt-20162 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_all_notes_150000_finetuned