
import io
from pathlib import Path

import imblearn
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "credit_card_fraud_smote_extra_trees.joblib"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ============================================================
# 2. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


# ============================================================
# 3. HEADER
# ============================================================

st.title("💳 Credit Card Fraud Detection")
st.write(
    "Detect potentially fraudulent credit-card transactions "
    "using a trained Extra Trees classification model."
)

st.caption(
    "Model: Extra Trees + SMOTE | "
    "Input: Time, V1–V28, Amount"
)

st.divider()


# ============================================================
# 4. MODEL FILE CHECK
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found:\n\n{MODEL_PATH}\n\n"
        "Place the saved .joblib model in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.info(
        "Ensure imbalanced-learn and scikit-learn are installed "
        "in the same Python environment that runs Streamlit."
    )
    st.stop()


# ============================================================
# 5. FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your transaction CSV",
    type=["csv"],
    help="Upload sample_transactions.csv or another CSV with the 30 required features.",
)

if uploaded_file is None:
    st.info(
        "Upload a CSV to begin. The Class column is optional; "
        "the model uses only the 30 input features."
    )

    with st.expander("Required input columns"):
        st.code(", ".join(FEATURES))

    st.stop()


# ============================================================
# 6. READ CSV ROBUSTLY
# ============================================================

st.write(f"**Uploaded file:** `{uploaded_file.name}`")
st.write(f"**File size:** {uploaded_file.size:,} bytes")

raw_bytes = uploaded_file.getvalue()

if not raw_bytes:
    st.error("The uploaded file is empty.")
    st.stop()

try:
    # Parse the uploaded bytes directly, avoiding stale file handles.
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        encoding="utf-8-sig",
        sep=",",
    )
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.code(repr(raw_bytes[:300]), language="text")
    st.stop()

# Normalize common column-name issues.
df.columns = (
    df.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

# Diagnose the actual file received by Streamlit.
with st.expander("Debug uploaded CSV"):
    st.write("Detected shape:", df.shape)
    st.write("Detected columns:", df.columns.tolist())
    st.write("First bytes of file:")
    st.code(repr(raw_bytes[:300]), language="text")
    st.dataframe(df.head(5), use_container_width=True)

if df.empty:
    st.error("The CSV contains no data rows.")
    st.stop()

if df.columns.duplicated().any():
    duplicates = df.columns[df.columns.duplicated()].tolist()
    st.error(f"Duplicate column names found: {duplicates}")
    st.stop()


# ============================================================
# 7. VALIDATE REQUIRED FEATURES
# ============================================================

missing = [column for column in FEATURES if column not in df.columns]

if missing:
    st.error("The uploaded CSV does not match the model's input schema.")
    st.write("**Missing required columns:**", missing)
    st.write("**Columns actually received:**", df.columns.tolist())

    if {"svg", "text", "category"}.issubset(set(df.columns)):
        st.warning(
            "This upload appears to contain SVG/text/category data, "
            "not credit-card transaction data. Check the filename shown "
            "above and select sample_transactions.csv from your project."
        )

    st.markdown(
        """
        **Expected:** `Time`, `V1`, `V2`, …, `V28`, `Amount`

        In the file picker, select the actual `sample_transactions.csv`
        file. Do not select an SVG, image, or unrelated export.
        """
    )
    st.stop()


# ============================================================
# 8. PREPARE MODEL INPUT
# ============================================================

X = df[FEATURES].copy()

for column in FEATURES:
    X[column] = pd.to_numeric(X[column], errors="coerce")

invalid_rows = X.isna().any(axis=1)

if invalid_rows.any():
    st.error(
        f"{int(invalid_rows.sum())} row(s) contain missing or "
        "non-numeric values in required feature columns."
    )
    st.dataframe(
        df.loc[invalid_rows].head(10),
        use_container_width=True,
    )
    st.stop()

if not np.isfinite(X.to_numpy(dtype=float)).all():
    st.error("The input contains infinite values. Correct the CSV and retry.")
    st.stop()

st.success(
    f"CSV validated successfully: {len(X):,} transactions, "
    f"{len(FEATURES)} features."
)


# ============================================================
# 9. DATA OVERVIEW
# ============================================================

a, b, c = st.columns(3)
a.metric("Transactions", f"{len(X):,}")
b.metric("Model Features", len(FEATURES))
c.metric("Invalid Feature Rows", "0")

with st.expander("Preview input transactions"):
    st.dataframe(df.head(10), use_container_width=True)


# ============================================================
# 10. PREDICTION
# ============================================================

if st.button("🔍 Detect Fraud", type="primary", use_container_width=True):
    try:
        predictions = np.asarray(model.predict(X)).astype(int)

        if not hasattr(model, "predict_proba"):
            st.error("The saved model does not support probability predictions.")
            st.stop()

        probabilities = model.predict_proba(X)
        model_classes = list(model.classes_)

        if 1 not in model_classes:
            st.error("The saved model does not contain fraud class 1.")
            st.stop()

        fraud_index = model_classes.index(1)
        fraud_probabilities = probabilities[:, fraud_index]

        results = df.copy()
        results["Predicted_Class"] = predictions
        results["Fraud_Probability"] = fraud_probabilities
        results["Prediction"] = np.where(
            predictions == 1,
            "Potential Fraud",
            "Legitimate",
        )

        st.session_state["fraud_detection_results"] = results

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# ============================================================
# 11. DISPLAY PREDICTIONS
# ============================================================

results = st.session_state.get("fraud_detection_results")

if results is not None:
    st.divider()
    st.header("📊 Prediction Summary")

    total = len(results)
    frauds = int((results["Predicted_Class"] == 1).sum())
    legitimate = total - frauds
    rate = (frauds / total * 100) if total else 0

    a, b, c, d = st.columns(4)
    a.metric("Total Transactions", f"{total:,}")
    b.metric("Potential Fraud", f"{frauds:,}")
    c.metric("Legitimate", f"{legitimate:,}")
    d.metric("Predicted Fraud Rate", f"{rate:.2f}%")

    st.subheader("🚨 Potentially Fraudulent Transactions")

    fraud_results = results[
        results["Predicted_Class"] == 1
    ].sort_values(
        "Fraud_Probability",
        ascending=False,
    )

    if fraud_results.empty:
        st.success("No transactions were classified as potentially fraudulent.")
    else:
        st.dataframe(fraud_results, use_container_width=True)

    st.subheader("All Predictions")
    st.dataframe(results, use_container_width=True)

    csv_output = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Prediction Results",
        data=csv_output,
        file_name="fraud_detection_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "Predictions are model estimates, not proof of fraud. "
        "Review flagged transactions before taking action."
    )



# Inspect model patterns
st.subheader("🔬 Why was this transaction flagged?")

fraud_rows = results[results["Predicted_Class"] == 1]

if fraud_rows.empty:
    st.info("No transactions were flagged as fraud.")
else:
    selected_index = st.selectbox(
        "Select a flagged transaction",
        fraud_rows.index.tolist(),
        format_func=lambda i: f"Transaction row {i + 1}"
    )

    st.write("Fraud probability:")
    st.metric(
        "Estimated probability",
        f"{results.loc[selected_index, 'Fraud_Probability']:.2%}"
    )

    st.write("Feature values for this transaction:")
    st.dataframe(
        results.loc[[selected_index], FEATURES].T.rename(
            columns={selected_index: "Feature Value"}
        ),
        use_container_width=True
    )

# Global feature importance
st.subheader("📊 Which features matter most to the model?")

try:
    estimator = model

    # If model is a pipeline, inspect its final estimator.
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": estimator.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.bar_chart(
            importance.head(10).set_index("Feature")["Importance"]
        )

        st.dataframe(importance.head(10), use_container_width=True)

        st.caption(
            "These are global feature importances across the model. "
            "They do not prove why one specific transaction was flagged."
        )
    else:
        st.info("Feature importance is unavailable for this estimator.")

except Exception as exc:
    st.warning(f"Could not retrieve feature importance: {exc}")

