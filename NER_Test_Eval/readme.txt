## Clinical bert - bert_pretrain_output_all_notes_150000  
### Extract Output Files on Test Set

python run_ner.py --do_train=false --do_eval=true --do_predict=true --vocab_file=../model/bert_pretrain_output_all_notes_150000/vocab.txt --bert_config_file=../model/bert_pretrain_output_all_notes_150000/bert_config.json --do_lower_case=False --init_checkpoint=../output/NER/bert_pretrain_output_all_notes_150000_finetuned/model.ckpt-99 --data_dir=../datasets/NER/ --output_dir=../output/NER/bert_pretrain_output_all_notes_150000_finetuned
