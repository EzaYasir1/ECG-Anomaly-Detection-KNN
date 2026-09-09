# ECG Anomaly Detection (KNN)

This project is a simplified reimplementation of [andturken/ECG-Anomaly-Detection](https://github.com/andturken/ECG-Anomaly-Detection), reducing a 5-class, 3-model pipeline down to a single K-Nearest Neighbors classifier distinguishing **Normal** vs **Abnormal** heartbeats, built to deepen my understanding of the full Machine learning pipeline (signal preprocessing, patient-level data leakage, hyperparameter tuning, and error analysis).

## Results

Trained and evaluated on the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (103,504 heartbeats across 44 patients), with a **patient-independent train/test split** to prevent data leakage.

|Metric|Normal|Abnormal|
|-|-|-|
|Precision|0.77|0.44|
|Recall|0.82|0.36|
|F1-score|0.80|0.40|

**Macro F1: 0.60**  |  **Accuracy: 0.69**  |  Best params: k=15, weights=uniform



!\[Confusion Matrix](Images/confusion\_matrix.png)

!\[Per-class metrics](Images/class\_metrics.png)





### Comparison to the original repo

||Model(s)|Classes|Split|Macro F1|
|-|-|-|-|-|
|Original repo|KNN, LogReg, Balanced RF|5 (N, A, L, R, V)|Cross-patient-group|0.48|
|**This project**|**KNN only**|**2 (Normal / Abnormal)**|**Patient-independent**|**0.60**|





## Error analysis

!\[Error analysis](Images/error\_analysis.png)



Comparing correctly classified vs missed Abnormal beats shows the model reliably catches abnormal beats with a sharp, deep post-QRS trough, but consistently misses a subset whose shape (shallower trough, taller trailing wave) overlaps with Normal morphology. The five individually-plotted false negatives are strikingly similar to each other, suggesting one specific abnormality subtype is systematically confused with Normal beats.

!\[Average waveforms](Images/average\_waveforms.png)
!\[K selection curve](Images/k\_selection\_curve.png)

## Changes made vs the original repo

* **Labels**: collapsed the original 5 classes (`N`, `A`, `L`, `R`, `V`) into 2 (Normal / Abnormal)
* **Model**: used KNN only
* **Split**: patient-independent (`GroupShuffleSplit`) so no patient's beats appear in both train and test — the original leaky,beat-level split gave a misleading 99% weighted F1 before this fix
* **Hyperparameter tuning**: grid search over k (1–20) and neighbor weighting (`uniform` vs `distance`), optimized on macro F1 rather than accuracy/weighted F1, since the latter hides poor minority-class performance in an imbalanced dataset (\~76% Normal / 24% Abnormal)
* Signal processing inspired from the original



## Tech stack

Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, Kaggle API, Google Colab

## Setup

```bash
pip install -r requirements.txt
```

Get the MIT-BIH dataset (Kaggle mirror, pre-converted to CSV):

```bash
pip install kaggle
export KAGGLE\\\_API\\\_TOKEN=your\\\_token\\\_here
kaggle datasets download -d mondejar/mitbih-database
unzip mitbih-database.zip -d mit\\\_bih\\\_csv
```

## Run

```bash
python download\\\_preprocess\\\_ecg.py   # signal processing -> data\\\_ecg.pickle
python classify\\\_ecg\\\_knn\\\_binary.py   # binary KNN training + evaluation
python plot\\\_ecg\\\_results.py          # generates the plots above
```

## Repo structure

```
├── download\\\_preprocess\\\_ecg.py   # signal processing 
├── classify\\\_ecg\\\_knn\\\_binary.py   # binary labels, patient split, KNN training/eval
├── plot\\\_ecg\\\_results.py          # confusion matrix, metrics, k-curve, error analysis
├── results\\\_log.json             # logged run history (metrics and confusion matrix per run)
├── requirements.txt
└── \\\*.png                        # saved result plots
```

## Future work

* **Address class imbalance directly**, try SMOTE (oversampling) or class-weighted variants on the training set to see if Abnormal recall improves without a comparable loss in Normal precision.
* **Cross-validated confidence intervals** , report macro F1 across multiple patient-group splits rather than a single held-out split, to see how stable the result is
* **Feature engineering** , extract clinically meaningful features (QRS width, R-R interval, P-wave presence) instead of using raw waveform points directly, and compare against the current raw-amplitude approach.
* **Deploy a minimal demo**,  a small Streamlit/Gradio app that takes a heartbeat segment and returns Normal/Abnormal, as a way to make the model interactively explorable.



## Acknowledgments

Built on top of the signal preprocessing from [andturken/ECG-Anomaly-Detection](https://github.com/andturken/ECG-Anomaly-Detection). Dataset: [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), PhysioNet.

