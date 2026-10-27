ED-predictor

Predict the direction of the next US Federal Reserve interest rate decision (up, down or no change) from the language in the Fed's own communications.

Upload a Fed document as a PDF (for example an FOMC statement or minutes) and the API returns the model's prediction for the next rate move.

How it works
Data: about 10,000 text passages from FOMC statements and minutes (2015–2023), paired with the 72 US Fed rate decisions in the same period.
Pairing with a sliding window: for each rate decision, all statements and minutes published since the previous decision are grouped together, so the model sees the communication that led up to that decision. Statements and minutes are handled separately because they are released on different schedules.
Text embeddings: each group of text is encoded with FinBERT, a BERT model pre-trained on financial language. Statement and minutes embeddings are concatenated into a single 1,536-dimension vector.
Model: a bidirectional LSTM (PyTorch) classifies the vector into up / down / no change. Class weighting compensates for "no change" being much more common than moves.
Training and model registry: 5-fold cross-validation; each run's model and metrics are saved with a timestamp, and the best-scoring model is automatically promoted to the checkpoint the API serves.
API: a FastAPI service accepts a PDF, extracts its text, runs it through FinBERT and the BiLSTM, and returns the predicted class.
Results
Model	Approach	Score
Baseline	TF-IDF + Multinomial Naive Bayes	65% accuracy, 0.60 macro F1 (cross-validated)
Final	FinBERT embeddings + BiLSTM	58% average cross-validation accuracy, 80% on the held-out test set

The dataset is small (72 decisions), so these figures vary noticeably between folds and should be read as indicative rather than definitive. See the notebooks for the full experiments.

Project structure
FED-predictor/
├── FED-predictor/
│   ├── api/fast.py          # FastAPI app (/ and /predict)
│   ├── interface/main.py    # Pipeline entry points: load, preprocess, train, predict
│   ├── ml_logic/
│   │   ├── data.py          # Loading, cleaning and the sliding-window pairing
│   │   ├── encoders.py      # FinBERT embeddings, splitting, PDF text extraction
│   │   ├── model.py         # BiLSTM, training with k-fold CV, evaluation
│   │   └── registry.py      # Saving/loading models and promoting the best one
│   └── params.py            # Hyperparameters and paths
├── notebooks/               # Baseline and deep-learning experiments
├── data/                    # Raw Fed texts and rate history
└── training_outputs/        # Saved models and metrics per training run