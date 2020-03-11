# W266 Final Project - Natural Language Processing 

## Brent Biseda and Katie Mo 
## Spring 2020 

# Methods

Training was performed using a 2080 TI on a desktop machine.  Each individual BERT model was trained for a total of 4 epochs and took approximately 1 hour per epoch.  In comparison, ELMO was trained for 5 epochs for a total of 10.9 hours or 2.2 hours per epoch.  

# Results - Comparison of Number of Epochs of Training for Test Set Accuracy

|Model                       |1 Epoch       |2 Epochs      |3 Epochs      |4 Epochs      |
|:---------------------------|-------------:|-------------:|-------------:|-------------:|
|Naive Bayes                 |              |              |              |              |
|Elmo                        |0.709426031   |              |              |              |
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
|Naive Bayes                 |             |             |             |             |
|Elmo                        |             |             |             |             |
|Bert Cased                  |0.4404324    |0.42128935   |0.44768453   |0.5104096    |
|Bert Un-Cased               |0.48347914   |0.49602044   |0.62805533   |0.73746926   |
|Biobert 1.0                 |0.44422722   |0.41489387   |0.4453975    |0.5038793    |
|Biobert 1.1                 |0.441695     |0.416389     |0.44800264   |0.4922997    |
|Clinical Bert (All Notes)   |0.44471493   |0.41527405   |0.45866716   |0.5149438    |
|Clinical Bert (Discharge)   |0.44352296   |0.4136234    |0.45590347   |0.52854675   |
|Clinical Biobert (All Notes)|0.44597864   |0.41869983   |0.45165843   |0.52100116   |
|Clincial Biobert (Discharge)|0.4440357    |0.41579548   |0.46043766   |0.53031653   |

# Results - 1 Epochs of Training

|Model                       |Test Accuracy |Test Loss    |
|:---------------------------|-------------:|------------:|
|Naive Bayes                 |              |             |
|Elmo                        |0.709426031   |             |
|Bert Cased                  |0.8237548     |0.4404324    |
|Bert Un-Cased               |0.8048023     |0.48347914   |
|Biobert 1.0                 |0.8241082     |0.44422722   |
|Biobert 1.1                 |0.8236432     |0.441695     |
|Clinical Bert (All Notes)   |0.8210393     |0.44471493   |
|Clinical Bert (Discharge)   |0.8240152     |0.44352296   |
|Clinical Biobert (All Notes)|0.8217833     |0.44597864   |
|Clincial Biobert (Discharge)|0.8232154     |0.4440357    |

# Results - 2 Epochs of Training

|Model                       |Test Accuracy |Test Loss    |
|:---------------------------|-------------:|------------:|
|Naive Bayes                 |              |             |
|Elmo                        |0.709426031   |             |
|Bert Cased                  |0.8506305     |0.42128935   |
|Bert Un-Cased               |0.81986755    |0.49602044   |
|Biobert 1.0                 |0.8540899     |0.41489387   |
|Biobert 1.1                 |0.85418296    |0.416389     |
|Clinical Bert (All Notes)   |0.8540155     |0.41527405   |
|Clinical Bert (Discharge)   |0.8549455     |0.4136234    |
|Clinical Biobert (All Notes)|0.85462934    |0.41869983   |
|Clincial Biobert (Discharge)|0.8552989     |0.41579548   |

# Results - 3 Epochs of Training

|Model                       |Test Accuracy |Test Loss    |
|:---------------------------|-------------:|------------:|
|Naive Bayes                 |              |             |
|Elmo                        |0.709426031   |             |
|Bert Cased                  |0.8756463     |0.44768453   |
|Bert Un-Cased               |0.82433134    |0.62805533   |
|Biobert 1.0                 |0.87719005    |0.4453975    |
|Biobert 1.1                 |0.87691104    |0.44800264   |
|Clinical Bert (All Notes)   |0.8774504     |0.45866716   |
|Clinical Bert (Discharge)   |0.8744188     |0.45590347   |
|Clinical Biobert (All Notes)|0.87272626    |0.45165843   |
|Clincial Biobert (Discharge)|0.8758695     |0.46043766   |


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