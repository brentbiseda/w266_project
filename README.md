# W266 Final Project - Natural Language Processing 

## Brent Biseda and Katie Mo 
## Spring 2020 

# Methods

Training was performed using a 2080 TI on a desktop machine.  Each individual BERT model was trained for a total of 4 epochs and took approximately 1 hour per epoch.  In comparison, ELMO was trained for 5 epochs for a total of 10.9 hours or 2.2 hours per epoch.  

# Results - 4 Epochs of Training

|Model                       |Test Accuracy |Test Loss    |
|:---------------------------|-------------:|------------:|
|Naive Bayes                 |              |             |
|Elmo                        |0.709426031   |             |
|Bert Cased                  |0.88762414    |0.5104096    |
|Bert Un-Cased               |0.8406242     |0.73746926   |
|Biobert 1.0                 |0.88749397    |0.5038793    |
|Biobert 1.1                 |0.87720865    |0.4922997    |
|Clinical Bert (All Notes)   |0.8883867     |0.5149438    |
|Clinical Bert (Discharge)   |0.8887959     |0.52854675   |
|Clinical Biobert (All Notes)|0.8885355     |0.52100116   |
|Clincial Biobert (Discharge)|0.8883867     |0.53031653   |



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

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.1_pubmed/model.ckpt-1000000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.1_finetuned

## Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_1.1_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.1_finetuned

# Bio Bert 1.0

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.0_pubmed_pmc/biobert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.0_finetuned

## Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_1.0_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.0_finetuned

# Bert Cased

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/cased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/cased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/bert_cased_finetuned

## Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_cased_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_cased_finetuned

# Bert Un-Cased

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=True --init_checkpoint=../model/uncased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/bert_uncased_finetuned

## Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_uncased_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_uncased_finetuned

# Clinical Bert - Biobert_pretrain_output_disch_100000

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_disch_100000_finetuned

## Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_pretrain_output_disch_100000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_disch_100000_finetuned

# Clinical Bert - Biobert_pretrain_output_all_notes_150000

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_all_notes_150000_finetuned

## Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_pretrain_output_all_notes_150000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_all_notes_150000_finetuned

# Clinical Bert - bert_pretrain_output_disch_100000

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_disch_100000_finetuned

## Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_pretrain_output_disch_100000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_disch_100000_finetuned

# Clinical Bert - bert_pretrain_output_all_notes_150000

## Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_all_notes_150000_finetuned

## Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_pretrain_output_all_notes_150000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_all_notes_150000_finetuned