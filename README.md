# W266 Final Project - Natural Language Processing 

## Brent Biseda and Katie Mo 
## Spring 2020 

# Methods

Training was performed using a 2080 TI on a desktop machine.  Each individual BERT model was trained for a total of 4 epochs and took approximately 1 hour per epoch.  In comparison, ELMO was trained for 5 epochs for a total of 10.9 hours or 2.2 hours per epoch.  

# Results for Sentiment Analysis Task

# Results - Comparison of Baseline Results

|Model                         |Test Accuracy       |
|:-----------------------------|-------------------:|
|Most Common Class             |0.6016627608525834  |
|Naive Bayes                   |                    |
|Elmo with Logistic Regression |0.709426031         |
|Bert Cased with Logistic Reg  |0.719953130231001   |

# Results - Comparison of Number of Epochs of Training for Test Set Accuracy

|Model                       |1 Epoch       |2 Epochs      |3 Epochs      |4 Epochs      |
|:---------------------------|-------------:|-------------:|-------------:|-------------:|
|Bert Cased                  |0.8237548     |0.8506305     |0.8756463     |0.88762414    |
|Bert Un-Cased               |0.8048023     |0.81986755    |0.82433134    |0.8406242     |
|Biobert 1.0                 |0.8241082     |0.8540899     |0.87719005    |0.88749397    |
|Biobert 1.1                 |0.8236432     |0.85418296    |0.87691104    |0.87720865    |
|Clinical Bert (All Notes)   |0.8210393     |0.8540155     |0.8774504     |0.8883867     |
|Clinical Bert (Discharge)   |0.8240152     |0.8549455     |0.8744188     |0.8887959     |
|Clinical Biobert (All Notes)|0.8217833     |0.85462934    |0.87272626    |0.8885355     |
|Clincial Biobert (Discharge)|0.8232154     |0.8552989     |0.8758695     |0.8883867     |

# Results - Comparison of Number of Epochs of Training for Test Set Loss

|Model                       |1 Epoch      |2 Epochs     |3 Epochs     |4 Epochs     |
|:---------------------------|------------:|------------:|------------:|------------:|
|Bert Cased                  |0.4404324    |0.42128935   |0.44768453   |0.5104096    |
|Bert Un-Cased               |0.48347914   |0.49602044   |0.62805533   |0.73746926   |
|Biobert 1.0                 |0.44422722   |0.41489387   |0.4453975    |0.5038793    |
|Biobert 1.1                 |0.441695     |0.416389     |0.44800264   |0.4922997    |
|Clinical Bert (All Notes)   |0.44471493   |0.41527405   |0.45866716   |0.5149438    |
|Clinical Bert (Discharge)   |0.44352296   |0.4136234    |0.45590347   |0.52854675   |
|Clinical Biobert (All Notes)|0.44597864   |0.41869983   |0.45165843   |0.52100116   |
|Clincial Biobert (Discharge)|0.4440357    |0.41579548   |0.46043766   |0.53031653   |


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

# Results for Presence of ADR Task

# Results - Comparison of Baseline Results

|Model                         |Test Accuracy       |
|:-----------------------------|-------------------:|
|Most Common Class             |0.5003380662609872  |
|Naive Bayes                   |                    |
|Elmo with Logistic Regression |0.9418526031102096  |
|Bert Cased with Logistic Reg  |0.8695064232589588  |

# Results - Comparison of Number of Epochs of Training for Test Set Accuracy

|Model                       |1 Epoch       |2 Epochs      |3 Epochs      |4 Epochs      |
|:---------------------------|-------------:|-------------:|-------------:|-------------:|
|Bert Cased                  |0.9607843     |0.98242056    |0.9858012     |0.98444897    |
|Bert Un-Cased               |0.92156863    |0.9425287     |0.94861394    |0.9472617     |
|Biobert 1.0                 |0.67207575    |0.8586883     |0.9060176     |0.9249493     |
|Biobert 1.1                 |0.95807976    |0.97565925    |0.98512506    |0.98444897    |
|Clinical Bert (All Notes)   |0.94929004    |0.9682218     |0.97363085    |0.9810683     |
|Clinical Bert (Discharge)   |0.82082486    |0.9229209     |0.9296822     |0.93509126    |
|Clinical Biobert (All Notes)|0.93644357    |0.9661934     |0.979716      |0.979716      |
|Clincial Biobert (Discharge)|0.9682218     |0.98039216    |0.9864774     |0.9858012     |

# Results - Comparison of Number of Epochs of Training for Test Set Loss

|Model                       |1 Epoch      |2 Epochs     |3 Epochs     |4 Epochs     |
|:---------------------------|------------:|------------:|------------:|------------:|
|Bert Cased                  |0.120792754  |0.07645559   |0.06911303   |0.08288954   |
|Bert Un-Cased               |0.24454235   |0.27175215   |0.28206244   |0.32833406   |
|Biobert 1.0                 |0.59864074   |0.38905054   |0.29607335   |0.26248044   |
|Biobert 1.1                 |0.1334118    |0.09078065   |0.06900492   |0.07189651   |
|Clinical Bert (All Notes)   |0.18111174   |0.14730458   |0.12958822   |0.1094518    |
|Clinical Bert (Discharge)   |0.4253079    |0.25793418   |0.25625613   |0.25129098   |
|Clinical Biobert (All Notes)|0.2359992    |0.14199895   |0.09437823   |0.091347925  |
|Clincial Biobert (Discharge)|0.12100793   |0.08341649   |0.065141685  |0.07421713   |

#Prolonged Training on Clinical BERT Discharge (10 Epochs)

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=10 --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/SENT/ --output_dir=../output/long/bert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/long/bert_pretrain_output_disch_100000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/long/bert_pretrain_output_disch_100000_finetuned

# Twitter Binary ADR Data Set 

## Clinical Bert - bert_pretrain_output_all_notes_150000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/ADR/bert_pretrain_output_all_notes_150000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_all_notes_150000_finetuned

## Clinical Bert - bert_pretrain_output_disch_100000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/ADR/bert_pretrain_output_disch_100000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_disch_100000_finetuned

## Clinical Bert - Biobert_pretrain_output_all_notes_150000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/ADR/biobert_pretrain_output_all_notes_150000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_all_notes_150000_finetuned

## Clinical Bert - Biobert_pretrain_output_disch_100000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/ADR/biobert_pretrain_output_disch_100000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_disch_100000_finetuned

## Bert Un-Cased

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=True --init_checkpoint=../model/uncased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_uncased_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/ADR/bert_uncased_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_uncased_finetuned

## Bert Cased

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/cased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/cased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_cased_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/ADR/bert_cased_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_cased_finetuned

## Bio Bert 1.0

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.0_pubmed_pmc/biobert_model.ckpt --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.0_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/ADR/biobert_1.0_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.0_finetuned

## Bio Bert 1.1

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.1_pubmed/model.ckpt-1000000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.1_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/ADR/biobert_1.1_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.1_finetuned

# Drugs.com ADR Data Set Sentiment Analysis

## Bio Bert 1.1

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.1_pubmed/model.ckpt-1000000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.1_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_1.1_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.1_finetuned

## Bio Bert 1.0

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.0_pubmed_pmc/biobert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.0_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_1.0_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.0_finetuned

## Bert Cased

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/cased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/cased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/bert_cased_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_cased_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_cased_finetuned

## Bert Un-Cased

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=True --init_checkpoint=../model/uncased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/bert_uncased_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_uncased_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_uncased_finetuned

## Clinical Bert - Biobert_pretrain_output_disch_100000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_pretrain_output_disch_100000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_disch_100000_finetuned

## Clinical Bert - Biobert_pretrain_output_all_notes_150000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_pretrain_output_all_notes_150000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_all_notes_150000_finetuned

## Clinical Bert - bert_pretrain_output_disch_100000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_pretrain_output_disch_100000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_disch_100000_finetuned

## Clinical Bert - bert_pretrain_output_all_notes_150000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_pretrain_output_all_notes_150000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_all_notes_150000_finetuned

# Extract Embeddings:

python extract_features.py --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --layers=-1 --init_checkpoint=../output/biobert_pretrain_output_disch_100000_finetuned/model.ckpt-40324 --input_file=../datasets/ADR/train.tsv --output_file=../output/extracted_features/train_features.json