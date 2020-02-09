# W266 Final Project - Natural Language Processing 

## Brent Biseda and Katie Mo 
## Spring 2020 

# Installation Instructions  

## CUDA Install  

cuda_10.0.130_411.31_win10.exe

## Pip install  

pip install --upgrade pip
pip install -r requirements.txt

# How to use sentiment analysis with Bert  
# https://medium.com/southpigalle/how-to-perform-better-sentiment-analysis-with-bert-ba127081eda

# This Paper was done with this  
# https://github.com/HSLCY/ABSA-BERT-pair

# Setting up Tensorboard

tensorboard --logdir=../output/regularbert/  --port=8008

# This line needs to be added in the tensorflow runner
writer = tf.train.SummaryWriter("/tmp/test", sess.graph)

# Using BioBERT  

## Named Entity Recognition Evaluation  

python run_ner.py --do_train=false --do_eval=true --vocab_file=../model/biobert_large/vocab_cased_pubmed_pmc_30k.txt --bert_config_file=../model/biobert_large/bert_config_bio_58k_large.json --init_checkpoint=../model/biobert_large/bio_bert_large_1000k.ckpt --num_train_epochs=10.0 --data_dir=../datasets/NER/BC2GM --output_dir=../output/biobert/

# Using Clinical BERT  

# Named Entity Recognition Evaluation  

python run_ner.py --do_train=false --do_eval=true --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --init_checkpoint=../model/biobert_pretrain_output_all_notes_150000/model.ckpt-150000 --num_train_epochs=10.0 --data_dir=../datasets/NER/BC2GM --output_dir=../output/clinicalbert/

# Using Regular BERT  

# Named Entity Recognition Evaluation  

python run_ner.py --do_train=false --do_eval=true --vocab_file=../model/wwm_uncased_L-24_H-1024_A-16/vocab.txt --bert_config_file=../model/wwm_uncased_L-24_H-1024_A-16/bert_config.json --init_checkpoint=../model/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt --num_train_epochs=10.0 --data_dir=../datasets/NER/BC2GM --output_dir=../output/regularbert/