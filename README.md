# Credit Card Fraud Detection Using Machine Learning

An end-to-end machine learning project that identifies potentially fraudulent credit card transactions using **Extra Trees Classifier and SMOTE** to address severe class imbalance.

The project covers data exploration, preprocessing, imbalanced classification, model comparison, evaluation, model serialization, and deployment using Streamlit.

## Project Overview

Credit card fraud detection is challenging because fraudulent transactions represent a very small fraction of all transactions. A model that predicts every transaction as legitimate could achieve high accuracy while failing to identify fraud.

This project addresses the challenge by applying SMOTE to the training data, comparing multiple classification algorithms, and selecting a model using **Precision-Recall AUC (PR-AUC)**.

## Key Highlights

- Analyzed 284,807 credit card transactions.
- Identified severe class imbalance: only 492 transactions were fraudulent.
- Used stratified train-test splitting.
- Applied SMOTE only to the training data.
- Compared five machine learning algorithms.
- Selected Extra Trees with SMOTE based on validation PR-AUC.
- Evaluated performance using precision, recall, F1-score, PR-AUC, and a confusion matrix.
- Built an interactive Streamlit application for CSV-based fraud prediction.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | Model training and evaluation |
| Imbalanced-learn | SMOTE implementation |
| Matplotlib / Seaborn | Exploratory data analysis and visualization |
| Joblib | Model serialization |
| Streamlit | Interactive web application |
| Git and GitHub | Version control and project hosting |

## Dataset

**Source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains transactions made by European cardholders in September 2013.

- **Total transactions:** 284,807
- **Legitimate transactions:** 284,315
- **Fraudulent transactions:** 492
- **Features:** 30 input features
- **Target:** `Class`

The input features include:

- `Time`: Seconds elapsed between the transaction and the first transaction in the dataset.
- `V1`–`V28`: Anonymized numerical features obtained through a transformation of the original data.
- `Amount`: Transaction amount.
- `Class`: Target label, where `0` represents legitimate transactions and `1` represents fraudulent transactions.

The dataset contains no missing values. Duplicate records were identified during data exploration and retained for the experiments.

## Machine Learning Workflow

```text
Raw Credit Card Transaction Dataset
                 |
                 v
     Exploratory Data Analysis
                 |
                 v
       Data Validation
                 |
                 v
  Stratified Train-Test Split (80:20)
                 |
          +------+------+
          |             |
          v             v
     Training Set     Test Set
          |             |
          v             |
       Apply SMOTE      |
          |             |
          v             |
    Model Comparison    |
          |             |
          v             |
  Select Best Model     |
          |             |
          v             |
    Final Evaluation <---+
          |
          v
   Save Trained Model
          |
          v
   Streamlit Application
          |
          v
  Upload CSV and Predict
```

### 1. Exploratory Data Analysis

The dataset was analyzed to understand:

- Class distribution and fraud prevalence.
- Missing values and duplicate records.
- Numerical feature distributions.
- Differences between legitimate and fraudulent transactions.

The analysis showed that fraud represented approximately **0.173%** of all transactions.

### 2. Train-Test Split

The data was split using an 80:20 stratified split with `random_state=42`.

| Dataset | Legitimate | Fraud | Total |
|---|---:|---:|---:|
| Training | 227,451 | 394 | 227,845 |
| Testing | 56,864 | 98 | 56,962 |

Stratification preserved the approximate class distribution in both sets.

### 3. Handling Class Imbalance with SMOTE

SMOTE stands for **Synthetic Minority Over-sampling Technique**.

Rather than merely duplicating existing fraud samples, SMOTE generates synthetic minority-class samples by interpolating between a fraud sample and one of its nearest fraud neighbors.

The interpolation formula is:

\[
x_{\text{new}}=x_i+\lambda(x_{nn}-x_i)
\]

Where:

- \(x_i\) is an existing minority-class sample.
- \(x_{nn}\) is a selected minority-class neighbor.
- \(\lambda\) is a random value between 0 and 1.
- \(x_{\text{new}}\) is the synthetic sample.

SMOTE was applied **only to the training data**, preventing synthetic oversampling from contaminating the untouched test set.

| Training data | Legitimate | Fraud | Total |
|---|---:|---:|---:|
| Before SMOTE | 227,451 | 394 | 227,845 |
| After SMOTE | 227,451 | 227,451 | 454,902 |

SMOTE generated 227,057 synthetic fraud samples.

### 4. Model Comparison

The following algorithms were evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Extra Trees Classifier
5. Hist Gradient Boosting Classifier

The experiments included baseline models and models trained with SMOTE where applicable.

**Model selection metric:** Validation PR-AUC.

PR-AUC is particularly useful for imbalanced classification because it summarizes the precision-recall trade-off across classification thresholds.

| Model | Validation PR-AUC |
|---|---:|
| Extra Trees — Baseline | 0.8214 |
| Extra Trees + SMOTE | 0.8238 |

Extra Trees with SMOTE achieved the stronger validation PR-AUC among these two Extra Trees configurations and was selected for final evaluation.

## Final Model Performance

The selected Extra Trees model trained with SMOTE was evaluated on the original, imbalanced test set.

| Metric | Score |
|---|---:|
| Precision | 93.33% |
| Recall | 85.71% |
| F1-score | 89.36% |
| PR-AUC | 0.8759 |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| **Actual Legitimate** | 56,858 | 6 |
| **Actual Fraud** | 14 | 84 |

Interpretation:

- **True negatives:** 56,858 legitimate transactions correctly identified.
- **False positives:** 6 legitimate transactions incorrectly flagged as fraud.
- **False negatives:** 14 fraudulent transactions missed by the model.
- **True positives:** 84 fraudulent transactions correctly identified.

At the evaluated classification threshold, the model detected 84 of the 98 fraudulent transactions in the test set.

These results reflect performance on this dataset and split; they do not guarantee equivalent results on future real-world transactions.

## Streamlit Application

The project includes an interactive application that allows users to upload a CSV file and receive fraud predictions.

### Application Features

- Upload transaction data in CSV format.
- Validate required columns and numeric input values.
- Predict whether each transaction is potentially fraudulent.
- Display predicted classes and fraud probabilities.
- Summarize flagged transactions.
- Download prediction results as a CSV file.

### Required CSV Columns

The uploaded CSV must contain all 30 input features:

```text
Time, V1, V2, V3, V4, V5, V6, V7, V8, V9,
V10, V11, V12, V13, V14, V15, V16, V17, V18,
V19, V20, V21, V22, V23, V24, V25, V26, V27,
V28, Amount
```

The target column `Class` is not required for prediction.

## Project Structure

```text
credit-card-fraud-detection/
│
├── app.py
├── requirements.txt
├── README.md
├── credit_card_fraud_smote_extra_trees.joblib
└── sample_transactions.csv
```

Additional notebooks, exploratory analysis, or evaluation reports can be included if they are part of the repository.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version supported by the selected dependencies.
- Git.
- A Kaggle account if you want to download the original dataset and reproduce the training experiments.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
joblib
```

For reproducible deployment, use the exact compatible package versions from your working environment.

### 4. Run the Application

Ensure the trained model file is in the same project directory as `app.py`.

```bash
streamlit run app.py
```

Streamlit will display a local URL where you can open the application in your browser.

## Deployment

The application can be deployed using Streamlit Community Cloud.

1. Push the project files to a GitHub repository.
2. Ensure `app.py`, `requirements.txt`, and the trained model file are available in the repository.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/).
4. Connect your GitHub account.
5. Select the repository and `app.py` as the entry point.
6. Deploy the application.

**Important:** Check the model file size before pushing it to GitHub. GitHub blocks ordinary Git files larger than 100 MB. If your model exceeds that limit, use Git LFS or an appropriate model-hosting solution.

## Limitations and Responsible Use

- The `V1`–`V28` features are anonymized, so their individual business meanings are not directly interpretable.
- Synthetic SMOTE samples may not represent valid real-world transactions.
- The classification threshold affects the balance between false positives and missed fraud.
- The dataset represents historical transactions and may not reflect current fraud patterns.
- A production fraud detection system would require monitoring, retraining, security controls, and validation on more recent data.
- Predictions should support further investigation, not automatically be treated as definitive proof of fraud.

## Future Improvements

- Tune the classification threshold based on fraud investigation costs.
- Compare precision-recall curves and calibration across models.
- Evaluate feature scaling and alternative imbalance-handling techniques.
- Add model explainability and prediction-level diagnostics.
- Monitor model drift and evaluate performance on newer transaction data.
- Add automated tests and input validation for production deployment.

## Key Learnings

This project demonstrates how to approach a real-world machine learning challenge where accuracy alone is not sufficient.

The most important lessons were:

- Why class imbalance can make ordinary classification misleading.
- How SMOTE generates synthetic minority-class examples.
- Why data leakage must be avoided during preprocessing.
- Why PR-AUC, precision, and recall are useful fraud detection metrics.
- How to serialize a trained model and integrate it into a Streamlit application.

## Acknowledgements

Thanks to the maintainers of the Kaggle Credit Card Fraud Detection dataset for making the data available for learning and experimentation.

---

**Disclaimer:** This is an educational portfolio project, not a production-ready banking fraud prevention system.
