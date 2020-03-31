# Detect Presence of ADR Task  

# Twitter Binary ADR Data Set 

## Clinical bert - bert_pretrain_output_all_notes_150000

### Run Training for ADR

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/bert_pretrain_output_all_notes_150000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_all_notes_150000_finetuned

## Clinical bert - bert_pretrain_output_disch_100000

### Run Training for ADR

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../model/bert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/bert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/bert_pretrain_output_disch_100000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/bert_pretrain_output_disch_100000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_pretrain_output_disch_100000_finetuned

## Clinical bert - Biobert_pretrain_output_all_notes_150000

### Run Training for ADR

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_all_notes_150000/model.ckpt-150000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_all_notes_150000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_all_notes_150000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/biobert_pretrain_output_all_notes_150000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_all_notes_150000_finetuned

## Clinical bert - Biobert_pretrain_output_disch_100000

### Run Training for ADR

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../model/biobert_pretrain_output_disch_100000/model.ckpt-100000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_disch_100000_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false  --do_predict=true --vocab_file=../model/biobert_pretrain_output_disch_100000/vocab.txt --bert_config_file=../model/biobert_pretrain_output_disch_100000/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/biobert_pretrain_output_disch_100000_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_pretrain_output_disch_100000_finetuned

## bert Un-Cased

### Run Training for ADR

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=ADR --do_lower_case=True --init_checkpoint=../model/uncased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_uncased_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/uncased_L-12_H-768_A-12/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/bert_uncased_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_uncased_finetuned

## bert Cased

### Run Training for ADR

python run_classifier.py --do_train=true --num_train_epochs=4 --vocab_file=../model/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../model/cased_L-12_H-768_A-12/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../model/cased_L-12_H-768_A-12/bert_model.ckpt --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_cased_finetuned

### Prediction on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/bert_cased_finetuned/model.ckpt-739 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/bert_cased_finetuned

## Bio bert 1.0

### Run Training for ADR

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=ADR --init_checkpoint=../model/biobert_v1.0_pubmed_pmc/biobert_model.ckpt --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.0_finetuned

### Prediction on Test Set

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.0_pubmed_pmc/vocab.txt --bert_config_file=../model/biobert_v1.0_pubmed_pmc/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/biobert_1.0_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.0_finetuned

## Bio bert 1.1

### Run Training for ADR

python run_classifier.py --do_train=true --num_train_epochs=4 --do_lower_case=False --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=ADR --init_checkpoint=../model/biobert_v1.1_pubmed/model.ckpt-1000000 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.1_finetuned

### Prediction on Test Set  

python run_classifier.py --do_train=false --do_predict=true --vocab_file=../model/biobert_v1.1_pubmed/vocab.txt --bert_config_file=../model/biobert_v1.1_pubmed/bert_config.json --task_name=ADR --do_lower_case=False --init_checkpoint=../output/ADR/biobert_1.1_finetuned/model.ckpt-1479 --data_dir=../datasets/ADR/ --output_dir=../output/ADR/biobert_1.1_finetuned
