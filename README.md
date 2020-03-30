# W266 Final Project - Natural Language Processing 

## Brent Biseda and Katie Mo 
## Spring 2020 

# Methods

Training was performed using a 2080 TI on a desktop machine.  Each individual BERT model was trained for a total of 4 epochs and took approximately 1 hour per epoch.  In comparison, ELMO was trained for 5 epochs for a total of 10.9 hours or 2.2 hours per epoch.  

# Results for NER Analysis Task

# Results - Comparison of Baseline Results

|Model                         |Test Accuracy       |
|:-----------------------------|-------------------:|
|Most Common Class             |0.938               |
|Naive Bayes                   |                    |
|ELMo with Logistic Regression |0.938               |
|BERT Cased with Logistic Reg  |0.948               |

# Results - Comparison of Number of Epochs of Training for Test Set Accuracy

|Model                       |1 Epoch       |2 Epochs      |3 Epochs      |4 Epochs      |
|:---------------------------|-------------:|-------------:|-------------:|-------------:|
|BERT Cased                  |0.9903476     |0.9907439     |0.9906307     |0.99102694    |
|BERT Un-Cased               |0.98737544    |0.9908005     |0.9908571     |0.99057406    |
|BioBERT 1.0                 |0.9904608     |0.9907439     |0.9909703     |0.990942      |
|BioBERT 1.1                 |0.9902627     |0.9907722     |0.99088544    |0.9907722     |
|Clinical BERT (All Notes)   |0.9904608     |0.99091375    |0.9908005     |0.9924989     |
|Clinical BERT (Discharge)   |0.99057406    |0.99071556    |0.9907722     |0.99142325    |
|Clinical BioBERT (All Notes)|0.99051744    |0.9907722     |0.99091375    |0.9917912     |
|Clincial BioBERT (Discharge)|0.9904042     |0.99091375    |0.99091375    |0.9908005     |

# Results - Comparison of Number of Epochs of Training for Test Set Loss

|Model                       |1 Epoch      |2 Epochs     |3 Epochs     |4 Epochs     |
|:---------------------------|------------:|------------:|------------:|------------:|
|BERT Cased                  |0.05552838   |0.038011767  |0.040655505  |0.028892228  |
|BERT Un-Cased               |0.062448576  |0.039479088  |0.037969436  |0.04100074   |
|BioBERT 1.0                 |0.04938845   |0.03651569   |0.03652621   |0.031140856  |
|BioBERT 1.1                 |0.056436066  |0.044782937  |0.042487428  |0.03401301   |
|Clinical BERT (All Notes)   |0.053461507  |0.036061972  |0.036518622  |0.024716122  |
|Clinical BERT (Discharge)   |0.03896644   |0.036543943  |0.0359913    |0.029238742  |
|Clinical BioBERT (All Notes)|0.049856387  |0.04229529   |0.03812202   |0.030528314  |
|Clincial BioBERT (Discharge)|0.049564358  |0.036872145  |0.037577145  |0.030808203  |

# Run NER Analysis Task

## Clinical bert - Biobert_pretrain_output_all_notes_150000

### Run Training for Sentiment Analysis

python run_ner.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/NER/ --output_dir=../output/NER/biobert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_ner.py --do_train=false --do_eval=true --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --do_lower_case=False --init_checkpoint=../output/NER/biobert_pretrain_output_all_notes_150000_finetuned/model.ckpt-99 --data_dir=../datasets/NER/ --output_dir=../output/NER/biobert_pretrain_output_all_notes_150000_finetuned

## Clinical bert - Biobert_pretrain_output_disch_100000

### Run Training for Sentiment Analysis

python run_ner.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/NER/ --output_dir=../output/NER/biobert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_ner.py --do_train=false --do_eval=true --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --do_lower_case=False --init_checkpoint=../output/NER/biobert_pretrain_output_disch_100000_finetuned/model.ckpt-99 --data_dir=../datasets/NER/ --output_dir=../output/NER/biobert_pretrain_output_disch_100000_finetuned

## Clinical bert - bert_pretrain_output_all_notes_150000

### Run Training for Sentiment Analysis

python run_ner.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/NER/ --output_dir=../output/NER/bert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_ner.py --do_train=false --do_eval=true --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --do_lower_case=False --init_checkpoint=../output/NER/bert_pretrain_output_all_notes_150000_finetuned/model.ckpt-99 --data_dir=../datasets/NER/ --output_dir=../output/NER/bert_pretrain_output_all_notes_150000_finetuned

## Clinical bert - bert_pretrain_output_disch_100000

### Run Training for Sentiment Analysis

python run_ner.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/NER/ --output_dir=../output/NER/bert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_ner.py --do_train=false --do_eval=true --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --do_lower_case=False --init_checkpoint=../output/NER/bert_pretrain_output_disch_100000_finetuned/model.ckpt-99 --data_dir=../datasets/NER/ --output_dir=../output/NER/bert_pretrain_output_disch_100000_finetuned

## bert Un-Cased

### Run Training for Sentiment Analysis

python run_ner.py --do_train=true --num_train_epochs=4 --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --do_lower_case=True --init_checkpoint=../model/uncased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/NER/ --output_dir=../output/NER/bert_uncased_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_ner.py --do_train=false --do_eval=true --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --do_lower_case=False --init_checkpoint=../output/NER/bert_uncased_finetuned/model.ckpt-99 --data_dir=../datasets/NER/ --output_dir=../output/NER/bert_uncased_finetuned

## bert Cased

### Run Training for Sentiment Analysis

python run_ner.py --do_train=true --num_train_epochs=4 --vocab_file=../model/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/cased_L-12_H-768_A-12/bert_config.json --do_lower_case=False --init_checkpoint=../model/cased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/NER/ --output_dir=../output/NER/bert_cased_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_ner.py --do_train=false --do_eval=true --vocab_file=../model/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/cased_L-12_H-768_A-12/bert_config.json --do_lower_case=False --init_checkpoint=../output/NER/bert_cased_finetuned/model.ckpt-99 --data_dir=../datasets/NER/ --output_dir=../output/NER/bert_cased_finetuned

## Bio bert 1.0

### Run Training for Sentiment Analysis

python run_ner.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --init_checkpoint=../model/biobert_v1.0_pubmed_pmc/biobert_model.ckpt --data_dir=../datasets/NER/ --output_dir=../output/NER/biobert_1.0_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_ner.py --do_train=false --do_eval=true --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --do_lower_case=False --init_checkpoint=../output/NER/biobert_1.0_finetuned/model.ckpt-99 --data_dir=../datasets/NER/ --output_dir=../output/NER/biobert_1.0_finetuned

## Bio bert 1.1

### Run Training for NER Task

python run_ner.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json  --init_checkpoint=../model/biobert_v1.1_pubmed/model.ckpt-1000000 --data_dir=../datasets/NER/ --output_dir=../output/NER/biobert_1.1_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_ner.py --do_train=false --do_eval=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --do_lower_case=False --init_checkpoint=../output/NER/biobert_1.1_finetuned/model.ckpt-99 --data_dir=../datasets/NER/ --output_dir=../output/NER/biobert_1.1_finetuned

# Results for Sentiment Analysis Task

# Results - Comparison of Baseline Results

|Model                         |Test Accuracy       |
|:-----------------------------|-------------------:|
|Most Common Class             |0.6016627608525834  |
|Naive Bayes                   |                    |
|ELMo with Logistic Regression |0.709426031         |
|BERT Cased with Logistic Reg  |0.719953130231001   |

# Results - Comparison of Number of Epochs of Training for Test Set Accuracy

|Model                       |1 Epoch       |2 Epochs      |3 Epochs      |4 Epochs      |
|:---------------------------|-------------:|-------------:|-------------:|-------------:|
|BERT Cased                  |0.8237548     |0.8506305     |0.8756463     |0.88762414    |
|BERT Un-Cased               |0.8048023     |0.81986755    |0.82433134    |0.8406242     |
|BioBERT 1.0                 |0.8241082     |0.8540899     |0.87719005    |0.88749397    |
|BioBERT 1.1                 |0.8236432     |0.85418296    |0.87691104    |0.87720865    |
|Clinical BERT (All Notes)   |0.8210393     |0.8540155     |0.8774504     |0.8883867     |
|Clinical BERT (Discharge)   |0.8240152     |0.8549455     |0.8744188     |0.8887959     |
|Clinical BioBERT (All Notes)|0.8217833     |0.85462934    |0.87272626    |0.8885355     |
|Clincial BioBERT (Discharge)|0.8232154     |0.8552989     |0.8758695     |0.8883867     |

# Results - Comparison of Number of Epochs of Training for Test Set Loss

|Model                       |1 Epoch      |2 Epochs     |3 Epochs     |4 Epochs     |
|:---------------------------|------------:|------------:|------------:|------------:|
|BERT Cased                  |0.4404324    |0.42128935   |0.44768453   |0.5104096    |
|BERT Un-Cased               |0.48347914   |0.49602044   |0.62805533   |0.73746926   |
|BioBERT 1.0                 |0.44422722   |0.41489387   |0.4453975    |0.5038793    |
|BioBERT 1.1                 |0.441695     |0.416389     |0.44800264   |0.4922997    |
|Clinical BERT (All Notes)   |0.44471493   |0.41527405   |0.45866716   |0.5149438    |
|Clinical BERT (Discharge)   |0.44352296   |0.4136234    |0.45590347   |0.52854675   |
|Clinical BioBERT (All Notes)|0.44597864   |0.41869983   |0.45165843   |0.52100116   |
|Clincial BioBERT (Discharge)|0.4440357    |0.41579548   |0.46043766   |0.53031653   |

# Results - Clinical BERT Discharge 10 Epochs

|Clinical BERT Discharge 10 Epochs | Score       |
|:---------------------------------|-------------------:|
|Test Accuracy                     |0.9063163           |
|Test Loss                         |0.6932565           |

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
|Most Common Class             |0.5003              |
|Naive Bayes                   |                    |
|ELMo with Logistic Regression |0.828537            |
|BERT Cased with Logistic Reg  |0.827338            |

# Results - Comparison of Number of Epochs of Training for Test Set Accuracy

|Model                       |1 Epoch       |2 Epochs      |3 Epochs      |4 Epochs      |
|:---------------------------|-------------:|-------------:|-------------:|-------------:|
|BERT Cased                  |0.88968825    |0.91846526    |0.88968825    |0.76978415    |
|BERT Un-Cased               |0.9088729     |0.9088729     |0.9088729     |0.90767384    |
|BioBERT 1.0                 |0.86930454    |0.90407676    |0.90527576    |0.9064748     |
|BioBERT 1.1                 |0.911271      |0.9064748     |0.91007197    |0.9088729     |
|Clinical BERT (All Notes)   |0.9028777     |0.91247004    |0.91366905    |0.91247004    |
|Clinical BERT (Discharge)   |0.88968825    |0.9088729     |0.91247004    |0.9148681     |
|Clinical BioBERT (All Notes)|0.9064748     |0.91366905    |0.9160671     |0.91846526    |
|Clincial BioBERT (Discharge)|0.9148681     |0.9172662     |0.91846526    |0.9160671     |

# Results - Comparison of Number of Epochs of Training for Test Set Loss

|Model                       |1 Epoch      |2 Epochs     |3 Epochs     |4 Epochs     |
|:---------------------------|------------:|------------:|------------:|------------:|
|BERT Cased                  |0.67930365   |0.49061382   |0.6920256    |0.5363002    |
|BERT Un-Cased               |0.34789237   |0.5582952    |0.6619164    |0.67761356   |
|BioBERT 1.0                 |0.3680852    |0.45204344   |0.50790715   |0.55730605   |
|BioBERT 1.1                 |0.42697513   |0.5832462    |0.7240624    |0.7711926    |
|Clinical BERT (All Notes)   |0.43992132   |0.6081       |0.6554421    |0.7103066    |
|Clinical BERT (Discharge)   |0.33103922   |0.4287269    |0.5021085    |0.55249363   |
|Clinical BioBERT (All Notes)|0.41861635   |0.5948032    |0.72895783   |0.751109     |
|Clincial BioBERT (Discharge)|0.35678673   |0.5739808    |0.6313981    |0.69853044   |

#Prolonged Training on Clinical BERT Discharge (10 Epochs)

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=10 --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/SENT/ --output_dir=../output/long/bert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/long/bert_pretrain_output_disch_100000_finetuned/model.ckpt-100810 --data_dir=../datasets/SENT/ --output_dir=../output/long/bert_pretrain_output_disch_100000_finetuned

# Twitter Binary ADR Data Set 

## Clinical bert - bert_pretrain_output_all_notes_150000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/bert_pretrain_output_all_notes_150000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_all_notes_150000_finetuned

## Clinical bert - bert_pretrain_output_disch_100000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/bert_pretrain_output_disch_100000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_disch_100000_finetuned

## Clinical bert - Biobert_pretrain_output_all_notes_150000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/biobert_pretrain_output_all_notes_150000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_all_notes_150000_finetuned

## Clinical bert - Biobert_pretrain_output_disch_100000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/biobert_pretrain_output_disch_100000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_disch_100000_finetuned

## bert Un-Cased

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=ADR --do_lower_case=True --init_checkpoint=../model/uncased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_uncased_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/bert_uncased_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_uncased_finetuned

## bert Cased

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/cased_L-12_H-768_A-12/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../model/cased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_cased_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/bert_cased_finetuned/model.ckpt-739 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_cased_finetuned

## Bio bert 1.0

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=ADR --init_checkpoint=../model/biobert_v1.0_pubmed_pmc/biobert_model.ckpt --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.0_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/biobert_1.0_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.0_finetuned

## Bio bert 1.1

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=ADR --init_checkpoint=../model/biobert_v1.1_pubmed/model.ckpt-1000000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.1_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/biobert_1.1_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.1_finetuned

# Drugs.com ADR Data Set Sentiment Analysis

## Bio bert 1.1

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.1_pubmed/model.ckpt-1000000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.1_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_1.1_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.1_finetuned

## Bio bert 1.0

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=drug --init_checkpoint=../model/biobert_v1.0_pubmed_pmc/biobert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.0_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_1.0_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_1.0_finetuned

## bert Cased

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/cased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/cased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/bert_cased_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_cased_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_cased_finetuned

## bert Un-Cased

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=True --init_checkpoint=../model/uncased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/SENT/ --output_dir=../output/bert_uncased_finetuned

### Run Prediction for Sentiment Analysis on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_uncased_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_uncased_finetuned

## Clinical bert - Biobert_pretrain_output_disch_100000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_pretrain_output_disch_100000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_disch_100000_finetuned

## Clinical bert - Biobert_pretrain_output_all_notes_150000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/biobert_pretrain_output_all_notes_150000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/biobert_pretrain_output_all_notes_150000_finetuned

## Clinical bert - bert_pretrain_output_disch_100000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_pretrain_output_disch_100000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_disch_100000_finetuned

## Clinical bert - bert_pretrain_output_all_notes_150000

### Run Training for Sentiment Analysis

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=drug --do_lower_case=False --init_checkpoint=../output/bert_pretrain_output_all_notes_150000_finetuned/model.ckpt-40324 --data_dir=../datasets/SENT/ --output_dir=../output/bert_pretrain_output_all_notes_150000_finetuned

# Extract Embeddings:

python extract_features.py --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --layers=-1 --init_checkpoint=../output/biobert_pretrain_output_disch_100000_finetuned/model.ckpt-40324 --input_file=../datasets/ADR/train.tsv --output_file=../output/extracted_features/train_features.json