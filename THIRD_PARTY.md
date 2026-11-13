# Data and algorithm attribution

## MovieLens 1M

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems 5(4), Article 19.
https://doi.org/10.1145/2827872

Dataset: https://grouplens.org/datasets/movielens/1m/
Download: https://files.grouplens.org/datasets/movielens/ml-1m.zip
Read data/ml-1m/README after downloading. Raw data and fitted artifacts are excluded from this repository.
This is an educational research project; do not redistribute the raw dataset or use it commercially without permission.

## LightGBM

Guolin Ke et al. 2017. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
We use the published LightGBM library with a classification formulation for recommendation.
This reproduces an application of the method, not the original paper's experiments or claimed scores.

## LRURec

Zhenrui Yue, Yueqi Wang, Zhankui He, Huimin Zeng, Julian McAuley, Dong Wang. 2024.
Linear Recurrent Units for Sequential Recommendation. WSDM, 930–938.
https://doi.org/10.1145/3616855.3635760
Official code: https://github.com/yueqirex/LRURec
Pinned commit: e02a559bd8c42342f5c53d69cefea2589becb22c

The upstream repository is linked as a Git submodule; its model/lru.py is imported unmodified.
We supply our own data split, training windows, experiment runner, evaluation and UI.
No pretrained checkpoints are used. We do not claim ownership of, or relicense, upstream code.
The upstream checkout did not include a top-level license at the pinned revision; consult its authors before reuse beyond this coursework.

## Item-based collaborative filtering

Badrul Sarwar et al. 2001. Item-based collaborative filtering recommendation algorithms. WWW.
https://doi.org/10.1145/371920.372071
Our baseline uses cosine similarity over binary positive interactions and a top-neighbor graph.
Confirm that item-based collaborative filtering matches the group's Lecture 7 material before final submission.
