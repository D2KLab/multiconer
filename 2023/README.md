# D2KLab at SemEval-2023 Task 2: Leveraging T-NER to develop a fine-tuned multilingual model

We used [T-NER](https://github.com/asahi417/tner), an open-source Python library, for fine-tuning a transformer-based language model for named entity recognition. T-NER provides an easy-to-use interface that allows for rapid experimentation with different language models, training data, and evaluation metrics.

## Table of Contents
- [Datasets](#datasets)
- [Experiments](#experiments)
  - [Base model generation](#base-model-generation)
  - [Fine-tuned model generation](#fine-tuned-model-generation)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Datasets

We fine-tuned our language model on a diverse range of named entity recognition (NER) datasets, including the Multilingual Complex Named Entity Recognition (MultiCoNER) 2022 dataset, as well as other publicly available datasets:

|                               Dataset name                                 |   Nb. of entities  |  Nb. of entity types  |     Language    |  Year  |
|:--------------------------------------------------------------------------:|:------------------:|:---------------------:|:---------------:|:------:|
| [tner/tweetner7](https://huggingface.co/datasets/tner/tweetner7)           |       11,380       |          7            |     English     |  2022  |
| [tner/tweebank_ner](https://huggingface.co/datasets/tner/tweebank_ner)     |        3,550       |          4            |     English     |  2022  |
| [tner/mit_restaurant](https://huggingface.co/datasets/tner/mit_restaurant) |        9,181       |          8            |     English     |  2014  |
| [tner/wnut2017](https://huggingface.co/datasets/tner/wnut2017)             |        4,691       |          6            |     English     |  2017  |
| [tner/bionlp2004](https://huggingface.co/datasets/tner/bionlp2004)         |       22,402       |          5            |     English     |  2004  |
| [tner/ontonotes5](https://huggingface.co/datasets/tner/ontonotes5)         |       76,714       |          8            |     English     |  2006  |
| [tner/bc5cdr](https://huggingface.co/datasets/tner/bc5cdr)                 |       16,423       |          2            |     English     |  2016  |
| [tner/fin](https://huggingface.co/datasets/tner/fin)                       |        1,467       |          4            |     English     |  2015  |
| [tner/btc](https://huggingface.co/datasets/tner/btc)                       |        9,339       |          3            |     English     |  2016  |
| [tner/conll2003](https://huggingface.co/datasets/tner/conll2003)           |       20,744       |          3            |     English     |  2003  |
| [tner/wikiann](https://huggingface.co/datasets/tner/wikiann)               |         -          |          3            |  282 languages  |  2017  |
| [tner/multinerd](https://huggingface.co/datasets/tner/multinerd)           |       13,048       |         17            |   9 languages   |  2022  |
| [tner/wikineural](https://huggingface.co/datasets/tner/wikineural)         |         -          |         16            |   9 languages   |  2021  |

## Experiments

T-NER offers a hyper parameters search approach in order to find the best hyper parameters across a set of given values. By using this feature, we have setup several experiments in order to know how much adding more data can improve a NER model and see until when it stops improving. The set of hyperparameters in T-NER was the same for all the experiments:

* learning rate: `1e−4` − `5e−4` − `1e−5` − `5e−5` − `1e−6` − `5e−6`
* batch size: `8` - `16` - `32`
* CRF: with (`1`) - without (`0`)
* gradient accumulation: `1` - `2` - `4`
* weight decay: `0` - `1e−6` − `1e−7` − `1e−8`
* max gradient normalization: `0` - `5` - `10` - `15`
* learning rate warmup: `0` - `0.1` - `0.2` - `0.3`

The CRF parameter is for using a CRF layer on top of output embedding or not. The selected model for the experiments on English data was [DeBERTaV3-large](https://huggingface.co/microsoft/deberta-v3-large). The reason we have selected this model is because DeBERTaV3 is currently the state of the art encoder model on many downstream tasks (see https://paperswithcode.com/paper/debertav3-improving-deberta-using-electra).

All the experiments have been done on 2 RTX 3090 GPUs using the same hyperparameters.

### Base model generation

The goal of this experiment is to train a model over a large number of [NER datasets](#datasets). To replicate the 1st experiment, one can run those lines:

```bash
git clone https://github.com/asahi417/tner
cd tner
tner-train-search -m "microsoft/deberta-v3-large" -c "1st_exp" -d "tner/tweetner7" " "tner/tweebank_ner" "tner/mit_restaurant" "tner/bionlp2004" "tner/wnut2017" "tner/mit_movie_trivia" "tner/ontonotes5" "tner/bc5cdr" "tner/fin" "tner/btc" "tner/conll2003" -e 15 --epoch-partial 5 --n-max-config 3 -b 64 -g 1 2 --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
```

The best trained model will be stored in a folder `1st_exp`.

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

### Fine-tuned model generation

We used the model from the previous experiment as a pre-tained model to finetune for our experiments on the MulticoNER 2023 dataset. The final model was trained with the same hyper parameters search values than the pre-trained model and finally the best hyper parameters stay the same as well. To replicate the 2nd experiment, one can run those lines:

```bash
tner-train-search -m "microsoft/deberta-v3-large" -c "2nd_exp" -d "tner/tweetner7" "tner/tweebank_ner" "tner/mit_restaurant" "tner/bionlp2004" "tner/wnut2017" "tner/mit_movie_trivia" "tner/ontonotes5" "tner/bc5cdr" "tner/fin" "tner/btc" "tner/conll2003" -l '{"train": "/data/multiconer/datasets/2022/EN-English/en_train.conll", "validation": "/data/multiconer/datasets/2022/EN-English/en_dev.conll", "test": "/data/multiconer/datasets/2022/EN-English/en_test.conll"}' -e 15 --epoch-partial 5 --n-max-config 3 -b 32 -g 1 2 --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
```

The best trained model will be stored in a folder `2nd_exp/best_model`.

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

## Evaluation

Our system generated a model that participated in all MultiCoNER tracks, with macro-averaged F1 being the official ranking metric. The final ranking is also available at https://multiconer.github.io/results.

|      Method       |  BN   |   DE   |  EN   |  ES   |  FA   |  FR   |  HI   |  IT   |  PT   |  SV   |  UK   |  ZH   |  Avg. |
|:-----------------:|:-----:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Official Baseline | 1.07  | 64.61  | 36.97 | 49.07 | 41.28 | 41.39 | 2.89  | 43.13 | 39.85 | 69.22 | 62.08 | 48.46 | 41.67 |
|       D2KLab      | 61.43 | 67.09  | 61.29 | 63.17 | 54.2  | 64.09 | 63.29 | 64.77 | 60.79 | 62.98 | 64.14 | 54.92 | 61.84 |

## Citation

Not available yet.
