# CineMatch · CS5481 Movie Recommendation

A working MovieLens 1M recommendation project with **exactly three algorithms**: ItemCF, LightGBM and LRURec (WSDM 2024). The Streamlit app recommends unseen movies, compares algorithms, explores the data and displays measured evaluation results.

## 本机启动（VS Code）

1. 在 VS Code 打开此项目文件夹。
2. 选择 Python 解释器：D:/PythonEnvs/cs5481_pro/Scripts/python.exe。
3. 按 F5，选择 **Movie recommendation app**；或在项目终端运行：

    & 'D:/PythonEnvs/cs5481_pro/Scripts/python.exe' -m streamlit run app.py

4. 浏览器打开 http://127.0.0.1:8501 。选择用户、算法、推荐数量及类型。

本机已训练权重放在 artifacts/。公开仓库不包含原始数据或模型缓存；其他电脑需按下方步骤复现。数据集并非现代电影目录，因此推荐的是 MovieLens 1M 收录的历史影片。

## Reproduce on another machine

Use Python 3.12 and Git. Clone with submodules:

    git clone --recurse-submodules https://github.com/winged-hussar-chat/cs5481_pro.git
    cd cs5481_pro
    python -m venv .venv

Activate the environment (Windows PowerShell: .venv/Scripts/Activate.ps1), then install:

    python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r requirements.txt
    python -m src.train --epochs 12
    python -m pytest -q
    python -m streamlit run app.py

The CUDA wheel is for supported NVIDIA GPUs. For CPU-only machines install the appropriate CPU PyTorch wheel instead; the runner automatically selects CPU if CUDA is unavailable. No standalone CUDA Toolkit is needed for this configuration. The measured Windows environment is recorded in requirements-lock.txt; to recreate it, install that file with --extra-index-url https://download.pytorch.org/whl/cu128. Platform-specific lock versions may need adjustment on other operating systems.

If the submodule is missing: git submodule update --init --recursive.
Training downloads MovieLens from GroupLens if absent. Read the downloaded data/ml-1m/README and comply with its terms. Model caches are trusted local files; do not load joblib files from unknown sources.

## What is implemented

| Requirement | Implementation |
|---|---|
| Dataset shape, nulls, dtypes, unique values | results/dataset.json and results/data_dictionary.csv; Data tab |
| Exploration and distributions | Rating and genre distributions, density, split sizes; Data tab |
| Preprocessing | Stable movie IDs; binary positive labels; time split; missing/duplicate diagnostics |
| Lecture method | Item-based CF with cosine similarity and top-k neighbors; confirm against your lecture material |
| Reference method | LightGBM binary classification with historical and metadata features |
| Recent method | Official LRURec model, pinned submodule; trained from scratch |
| At least five unseen movies | Top 5–20 titles and genres; all observed ratings excluded in demo |
| Algorithm comparison | Three-column comparison, common-movie overlap, shared evaluation protocol |
| Interactive interface | Streamlit app; no remote hosting required |

The assignment phrase “comparison dataset” is interpreted here as a common train/validation/test benchmark and a model comparison table. Confirm this interpretation with the TA if a specific table is expected. A formal report and presentation are intentionally deferred; this repository focuses on the executable project.

## Evaluation protocol

- Fixed seed **42** for NumPy, Python, PyTorch and LightGBM.
- **Global chronological 80/10/10 split**, keeping equal timestamps together: 800,168 / 100,020 / 100,021 rows.
- Training cutoff: 2000-12-02 14:52:18 UTC; validation cutoff: 2000-12-29 23:42:56 UTC.
- Relevance is rating >= 4. Rating 1–3 is not a positive interaction, but all training-rated items are excluded from recommendations.
- Candidate universe: **3,662 training-catalog movies**, not a sampled set of 100 negatives. Same candidates and targets for all models.
- Metrics: macro-average Recall@10, NDCG@10 and HitRate@10 over users with at least one eligible held-out positive.
- **Warm-start only**: validation covers 492 of 1,103 positive-rating users; test covers 990 of 1,188. Cold users and unknown items are explicitly excluded; these scores do not describe cold-start quality.
- Validation chooses CF neighbors (40/80), LightGBM leaves (31/63), and LRURec checkpoint. Test is not used for feature, parameter or model selection.
- Models remain frozen after training. Validation interactions are not appended to test histories.
- Demo exclusion is stronger: all known ratings are hidden from recommendations, including later observations. That display rule does not feed training features or offline metrics.

### Algorithms and leakage controls

**ItemCF:** positive-interaction user-item matrix, item cosine similarity, top-neighbor graph. Scores sum neighbor similarities for the user's positive training history.

**LightGBM:** within the outer training period, the first 80% by time supplies fixed feature statistics and the final 20% supplies positive label rows. Features include item popularity, mean ratings, release year, genre indicators, user demographics, history length and genre preferences (45 features). Four sampled unrated items per positive are labeled negative. Sampling excludes outer-training-rated items and never inspects validation/test labels. This is a pointwise classifier, so query groups are not required. Unobserved items are not guaranteed dislikes; reported classifier scores are not calibrated satisfaction probabilities. Features remain frozen to the inner context at inference to avoid label-row self leakage.

**LRURec:** imports the official unmodified model/lru.py from the pinned submodule. Two blocks, hidden width 32, sequence length 64, batch 32, AdamW learning rate 0.001, weight decay 0.01, gradient clipping 5. Positive histories are split into next-item training windows; no cross-user transitions. Maximum 12 epochs, early stopping after three non-improving validation epochs. Best checkpoint is epoch 1 in this run. No pretrained weights. This is a small coursework adaptation, **not a reproduction of the paper's original benchmark scores**, which uses a different split/configuration.

## Measured results

See results/metrics.csv for exact values. Test set: 990 eligible users, full training-catalog ranking.

| Model | Recall@10 | NDCG@10 | HitRate@10 |
|---|---:|---:|---:|
| ItemCF | 0.05694 | 0.21683 | 0.62020 |
| LightGBM | 0.05064 | 0.19079 | 0.61313 |
| LRURec | 0.05194 | 0.16284 | 0.56061 |

ItemCF also has the highest validation NDCG@10 and is the default demo model. LRURec validation loss improves while ranking quality declines after epoch 1, so early stopping matters. Results are from one seed; small differences are not evidence of statistical significance. Global temporal splitting exposes cold-start limitations, and popular-item bias remains. Future experiments should be planned on validation data before another final test evaluation.

## Files and outputs

- app.py: bilingual interactive demo.
- src/data.py: download, exploration summaries, timestamp split, targets.
- src/models.py: ItemCF, feature construction, LightGBM and upstream model loading.
- src/train.py: tuning, training, checkpointing, full-catalog evaluation.
- src/service.py: demo inference and unseen-item filtering.
- tests/: ranking, temporal boundary, gradient, feature-scale and real inference checks.
- results/metrics.csv: common algorithm comparison table.
- results/training_history.csv: loss and validation ranking per epoch.
- results/tuning.json, results/protocol.json: trials, selected settings, timings, hardware and limitations.
- results/figures/: generated plots (run python -m src.analyze).
- data/, artifacts/: local only, ignored by Git.
- .vscode/: launch and test tasks; machine-specific interpreter settings stay local.

Hardware used: Ryzen 9 8940HX, RTX 5060 Laptop 8GB, Windows 11, Python 3.12.6, PyTorch 2.11.0+cu128. Exact measured runtime is in results/protocol.json; training time depends on hardware. GPU kernels can have small nondeterministic differences.

## Attribution

See THIRD_PARTY.md for MovieLens/GroupLens, LightGBM, ItemCF and LRURec citations and reuse boundaries. The public repository contains original project code, aggregate results and a pointer to upstream LRURec, not the raw MovieLens dataset.
