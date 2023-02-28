# D2KLab at SemEval Task 2: MultiCoNER

This repository contains the code used by the D2KLab team for [SemEval Task 2: MultiCoNER](https://multiconer.github.io/). The goal of this task is to detect semantically ambiguous and complex entities in short and low-context settings.

Each folder represents the year we have participated and the experiements we have conducted.

## Experiment Results

We evaluated our fine-tuned language model on the test sets of the MultiCoNER datasets. The results are summarized in the tables below:

### 2022

|      Method       |  BN   |   DE   |  EN   |  ES   |  FA   |  HI   |  KO   |  NL   |  RU   |  TR   |  ZH   |  MIX  | MULTI |  Avg. |
|:-----------------:|:-----:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Official Baseline | 39.41 | 63.74  | 61.36 | 57.84 | 52.24 | 48.22 | 55.25 | 62.01 | 59.59 | 46.25 | 63.4  | 58.14 | 48.22 | 47.78 |
|       D2KLab      | 52.57 | 67.09  | 74.57 | 62.77 | 55.91 | 52.78 | 64.96 | 66.7  | 68.21 | 56.57 | 54.92 | 77.6  | 52.78 | 68.08 |

### 2023

|      Method       |  BN   |   DE   |  EN   |  ES   |  FA   |  FR   |  HI   |  IT   |  PT   |  SV   |  UK   |  ZH   |  Avg. |
|:-----------------:|:-----:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Official Baseline | 1.07  | 64.61  | 36.97 | 49.07 | 41.28 | 41.39 | 2.89  | 43.13 | 39.85 | 69.22 | 62.08 | 48.46 | 41.67 |
|       D2KLab      | 61.43 | 67.09  | 61.29 | 63.17 | 54.2  | 64.09 | 63.29 | 64.77 | 60.79 | 62.98 | 64.14 | 54.92 | 61.84 |
