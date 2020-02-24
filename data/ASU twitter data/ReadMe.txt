This file contains information about the Twitter ADR data set (version 1.0) containing annotations for mentions of Adverse Drug Reactions (ADR) and Indications (the reson to use the drug).

NOTE

Please note that by downloading the Twitter data you agree to follow the Twitter terms of service (https://twitter.com/tos), which requires not to redistribute the data and to delete tweets that are marked deleted in the futur.
You MUST NOT re-distribute the tweets, the annotations or the corpus obtained, as this violates the Twitter Terms of Use.

DATA
The Twitter data set is divided into train and test sets. For each set the information about the tweets and the annotations are saved in separate files: Tweet ID files and annotation files .
Tweet ID Files: train_tweet_ids.tsv, test_tweet_ids.tsv

These files contain tab separated information about tweet IDs, user IDs and text IDs as shown in the example below. 

351772771174449152	632397611	fluoxetine-51d1cd8b53785f584a9af0b3

The tweet ID and user ID can be used by the Twitter API for downloading the tweets. The text ID links the tweet to the corresponding annotations in the annotation file.  


Annotation Files: train_tweet_annotations.tsv, test_tweet_annotations.tsv

These files contain tab separated information about the details of the annotations including: text ID, start offset, end offset, semantic type, annotated text, related drug and target drug. The following line is an example annotation line (the corresponding UMLS IDs will be released in the future versions of the annotations). Please note that the related drug is the drug that was used as a keyword in Twitter search query and the target drug is the drug that the current annotation (ADR or Indication) is targeting. Target drug can be different from the related drug in cases where there are more than one drug mentions in a tweet. 

fluoxetine-51d1cd8b53785f584a9af0b3	13	34	ADR	Restless Leg Syndrome	fluoxetine	fluoxetine


DOWNLOAD
The individual tweets can be obtained via the url:
 http://twitter.com/[userid]/status/[tweetid]
For example, for the sample line from the test_tweet_ids, the tweet can be accessed via the following url:

https://twitter.com/632397611/status/342043859523600384

In order to make the obtaining of the associated tweets easier for researchers, we provide a simple python script:

- download_tweets.py

This script requires python to be installed along with the beautifulsoup4 package.

beautifulsoup4 can be installed easily via easy_install:

easy_install beautifulsoup4

To run the script, please use the following command:

python download_tweets.py [input_filename] > [output_filename]

CONTACT:
Azadeh Nikfarjam, anikfarj@asu.edu

CITATION
Please cite the following paper when using this data:

@article {
	author = {Nikfarjam, Azadeh and Sarker, Abeed and O{\textquoteright}Connor, Karen and Ginn, Rachel and Gonzalez, Graciela},
	title = {Pharmacovigilance from social media: mining adverse drug reaction mentions using sequence labeling with word embedding cluster features},
	year = {2015},
	doi = {10.1093/jamia/ocu041},
	publisher = {The Oxford University Press},
	isbn = {1527-974X},
	issn = {1067-5027},
	journal = {Journal of the American Medical Informatics Association}
}



