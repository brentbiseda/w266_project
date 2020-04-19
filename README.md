# W266 Natural Language Processing - Final Project

## Enhancing Pharmacovigilance with Drug Reviews and Social Media

### Brent Biseda and Katie Mo
### Spring 2020

**Abstract**

This paper explores whether the use of drug reviews and social media could be leveraged as potential alternative sources for pharmacovigilance of adverse drug reactions (ADRs). We examined the performance of BERT (Devlin et al., 2018) alongside two variants that are trained on biomedical papers, BioBERT (Jinhyuk et al., 2019), and clinical notes, Clinical BERT (Alsentzer et al., 2019).  A variety of 8 different BERT models were fine-tuned and compared across three different tasks in order to evaluate their relative performance to one another in the ADR tasks. The tasks include sentiment classification of drug reviews, presence of ADR in twitter postings, and named entity recognition of ADRs in twitter postings. BERT demonstrates its flexibility with high performance across all three different pharmacovigilance related tasks.

**Tasks**

- [Sentiment Classification of Drug Reviews](./task1_sentiment)
- [ADR Classification of Tweets](./task2_adr)
- [Named entity recognition of ADRs in Tweets](./task3_ner)

**Deliverables**

- [Final Project Paper](https://github.com/brentbiseda/w266_project/blob/master/W266%20Final%20Project%20Paper.pdf)
- [Final Project Presentation](https://github.com/brentbiseda/w266_project/blob/master/W266%20Final%20Project%20Presentation%20Slides.pdf)

**Datasets**

- [Drug reviews](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29)
- [ADRs in Twitter](http://diego.asu.edu/Publications/ADRMine.html)

**Github References**

- [BERT](https://github.com/google-research/bert)
- [BioBERT](https://github.com/dmis-lab/biobert)
- [clinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT)
- [clinical-BioBERT](https://github.com/arnavbhandari/clinical-BioBERT)


## Installation    

#### CUDA Install  

cuda_10.0.130_411.31_win10.exe

#### Pip install  

pip install --upgrade pip
pip install -r requirements.txt

#### Start up Tensorboard and set directory

tensorboard --logdir ../output

Go to Tensorboard location: http://localhost:6006
