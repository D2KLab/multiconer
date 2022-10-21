# SemEval 2023 MultiCoNER experiments

All the experiments are done on Gravette on 2 GPUs.

## 1st experiment

The goal of this experiment is to train a model over a large number of NER datasets. The training is done with [T-NER](https://github.com/asahi417/tner). To replicate the 1st experiment, one can run those lines:
```
git clone https://github.com/asahi417/tner
cd tner
tner-train-search -m "microsoft/deberta-v3-large" -c "/data/muticoner2023/1st_exp/" -d "tner/tweetner7" " "tner/tweebank_ner" "tner/mit_restaurant" "tner/bionlp2004" "tner/wnut2017" "tner/mit_movie_trivia" "tner/ontonotes5" "tner/bc5cdr" "tner/fin" "tner/btc" "tner/conll2003" -e 15 --epoch-partial 5 --n-max-config 3 -b 64 -g 1 2 --lr 1e-6 1e-5 --crf 0 1 --max-grad-norm 0 10 --weight-decay 0 1e-7
```
The best trained model will be in `/data/multiconer2023/1st_exp/best_model`
