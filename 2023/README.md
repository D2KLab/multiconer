# SemEval 2023 MultiCoNER experiments

All the experiments are done on Gravette on 2 GPUs.

## 1st experiment

The goal of this experiment is to train a model over a large number of NER datasets. The training is done with [T-NER](https://github.com/asahi417/tner). To replicate the 1st experiment, one can run those lines:
```
git clone https://github.com/asahi417/tner
cd tner
tner-train-search -m "microsoft/deberta-v3-large" -c "/data/multiconer/models/2023/tner/1st_exp/" -d "tner/tweetner7" " "tner/tweebank_ner" "tner/mit_restaurant" "tner/bionlp2004" "tner/wnut2017" "tner/mit_movie_trivia" "tner/ontonotes5" "tner/bc5cdr" "tner/fin" "tner/btc" "tner/conll2003" -e 15 --epoch-partial 5 --n-max-config 3 -b 64 -g 1 2 --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
```
The best trained model will be in `/data/multiconer/models/2023/tner/1st_exp/best_model`

Results:
```
			    precision	        recall		f1-score	support
actor			    0.95		0.97		0.96		591
amenity			    0.66		0.71		0.68		292
award			    0.43		0.56		0.48		41
cardinal_number		    0.71		0.85		0.77		938
cell_line		    0.76		0.80		0.78		505
cell_type		    0.78		0.82		0.80		628
character_name		    0.69		0.75		0.72		104
chemical		    0.93		0.93		0.93		5347
corporation		    0.25		0.15		0.19		34
cuisine			    0.82		0.87		0.85		307
date			    0.78		0.88		0.83		1818
director		    0.77		0.92		0.84		206
disease			    0.79		0.85		0.82		4244
dish			    0.75		0.77		0.76		122
dna			    0.78		0.77		0.78		1260
event			    0.72		0.63		0.67		143
facility		    0.53		0.77		0.63		115
genre			    0.66		0.65		0.66		387
geopolitical_area	    0.89		0.92		0.91		2268
group			    0.83		0.87		0.85		886
language		    0.78		0.88		0.83		33
law			    0.70		0.75		0.72		40
location		    0.88		0.86		0.87		2815
money			    0.80		0.89		0.85		349
opinion			    0.29		0.51		0.37		87
ordinal_number		    0.77		0.83		0.80		232
organization		    0.85		0.86		0.86		3643
origin			    0.40		0.44		0.42		110
other			    0.78		0.82		0.80		1019
percent			    0.85		0.88		0.87		177
person			    0.93		0.93		0.93		5767
plot			    0.46		0.52		0.49		802
product			    0.54		0.33		0.41		186
protein			    0.80		0.88		0.84		3029
quantity		    0.73		0.80		0.76		100
quote			    0.76		1.00		0.87		13
rating			    0.67		0.82		0.74		83
relationship		    0.27		0.43		0.34		69
restaurant		    0.88		0.90		0.89		146
rna			    0.84		0.89		0.86		131
soundtrack		    0.67		0.40		0.50		5
time			    0.66		0.78		0.71		333
work_of_art		    0.51		0.39		0.44		247

micro avg		    0.83		0.86		0.84		39652
macro avg		    0.71		0.75		0.72		39652
```

## 2nd experiment

The goal of this experiment is to train a model over a large number of NER datasets, including MulticoNER 2022. To replicate the 2nd experiment, one can run those lines, still with T-NER:
```
tner-train-search -m "microsoft/deberta-v3-large" -c "/data/multiconer/models/2023/tner/2nd_exp/" -d "tner/tweetner7" "tner/tweebank_ner" "tner/mit_restaurant" "tner/bionlp2004" "tner/wnut2017" "tner/mit_movie_trivia" "tner/ontonotes5" "tner/bc5cdr" "tner/fin" "tner/btc" "tner/conll2003" -l '{"train": "/data/multiconer/datasets/2022/EN-English/en_train.conll", "validation": "/data/multiconer/datasets/2022/EN-English/en_dev.conll", "test": "/data/multiconer/datasets/2022/EN-English/en_test.conll"}' -e 15 --epoch-partial 5 --n-max-config 3 -b 32 -g 1 2 --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
```
The best trained model will be in `/data/multiconer/models/2023/tner/2nd_exp/best_model`

Results:
```
			    precision	        recall		f1-score	support
actor			    0.96		0.97		0.96		591
amenity			    0.66		0.69		0.67		292
award			    0.42		0.51		0.46		41
cardinal_number		    0.74		0.83		0.78		938
cell_line		    0.78		0.79		0.78		505
cell_type		    0.74		0.82		0.78		628
character_name		    0.71		0.81		0.75		104
chemical		    0.93		0.92		0.93		5347
corporation		    0.79		0.74		0.77		227
cuisine			    0.88		0.86		0.87		307
date			    0.79		0.86		0.82		1818
director		    0.78		0.92		0.84		206
disease			    0.84		0.84		0.84		4244
dish			    0.76		0.82		0.79		122
dna			    0.76		0.79		0.78		1260
event			    0.74		0.63		0.68		143
facility		    0.54		0.80		0.65		115
genre			    0.66		0.65		0.66		387
geopolitical_area	    0.89		0.92		0.91		2268
group			    0.83		0.88		0.85		1076
language		    0.80		0.85		0.82		33
law			    0.72		0.80		0.76		40
location		    0.89		0.87		0.88		3049
money			    0.83		0.91		0.86		349
opinion			    0.25		0.44		0.32		87
ordinal_number		    0.80		0.84		0.82		232
organization		    0.86		0.86		0.86		3643
origin			    0.41		0.42		0.41		110
other			    0.82		0.82		0.82		1019
percent			    0.89		0.90		0.90		177
person			    0.94		0.93		0.93		6057
plot			    0.52		0.55		0.53		802
product			    0.68		0.59		0.64		333
protein			    0.82		0.87		0.84		3029
quantity		    0.72		0.78		0.75		100
quote			    0.87		1.00		0.93		13
rating			    0.71		0.80		0.75		83
relationship		    0.36		0.54		0.43		69
restaurant		    0.95		0.90		0.93		146
rna			    0.79		0.85		0.82		131
soundtrack		    0.50		0.40		0.44		5
time			    0.71		0.74		0.72		333
work_of_art		    0.63		0.57		0.60		423

micro avg		    0.84		0.86		0.85		40882
macro avg		    0.74		0.77		0.75		40882
```
