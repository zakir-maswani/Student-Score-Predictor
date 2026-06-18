import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="📚",
    layout="wide",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Load dataset (optional – used for chart context & metrics) ────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("students_score.csv")
        return df
    except FileNotFoundError:
        return None

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 Score Predictor")
    st.markdown("Predict a student's exam score based on hours studied.")
    st.divider()

    hours = st.slider(
        "🕐 Hours Studied",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.5,
        help="Drag to set the number of hours the student studied.",
    )

    hours_input = st.number_input(
        "Or type exact hours:",
        min_value=0.0,
        max_value=12.0,
        value=hours,
        step=0.25,
    )

    # The number input takes priority if it differs from the slider
    study_hours = hours_input

    predict_btn = st.button("Predict Score", use_container_width=True, type="primary")

    st.divider()

    # Model metrics from the training data (if CSV is available)
    if df is not None:
        st.subheader("📊 Model Performance")
        X = df[["Hours"]]
        y = df["Scores"]

        # Re-derive metrics using the saved model on the full dataset
        y_pred_all = model.predict(X)
        r2 = r2_score(y, y_pred_all)
        mse = mean_squared_error(y, y_pred_all)
        mae = mean_absolute_error(y, y_pred_all)

        col1, col2 = st.columns(2)
        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("MAE", f"{mae:.2f}")
        st.metric("MSE", f"{mse:.2f}")
        st.caption("Metrics computed on the full dataset using the saved model.")

# ── Main content ──────────────────────────────────────────────────────────────
st.title("🎓 Student Exam Score Predictor")
st.markdown(
    "Enter the number of hours a student studied and get an instant prediction "
    "of their expected exam score using a **Linear Regression** model."
)
st.divider()

# Prediction
predicted_score = model.predict(np.array([[study_hours]]))[0]
predicted_score_clamped = float(np.clip(predicted_score, 0, 100))

# ── Metric cards ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="⏱️ Hours Studied",
        value=f"{study_hours:.2f} hrs",
    )

with col2:
    st.metric(
        label="🎯 Predicted Score",
        value=f"{predicted_score_clamped:.1f} / 100",
    )

with col3:
    # Grade band
    if predicted_score_clamped >= 90:
        grade, colour = "A", "🟢"
    elif predicted_score_clamped >= 75:
        grade, colour = "B", "🔵"
    elif predicted_score_clamped >= 60:
        grade, colour = "C", "🟡"
    elif predicted_score_clamped >= 45:
        grade, colour = "D", "🟠"
    else:
        grade, colour = "F", "🔴"

    st.metric(
        label="📋 Grade Band",
        value=f"{colour} Grade {grade}",
    )

st.divider()

# ── Regression chart ──────────────────────────────────────────────────────────
st.subheader("📈 Regression Line with Your Prediction")

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")

if df is not None:
    ax.scatter(
        df["Hours"],
        df["Scores"],
        color="#4C9BE8",
        alpha=0.65,
        s=55,
        label="Actual Data",
        zorder=2,
    )

    # Regression line
    x_line = np.linspace(df["Hours"].min(), df["Hours"].max(), 300).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color="#F5A623", linewidth=2.2, label="Regression Line", zorder=3)
else:
    # No CSV — draw regression line over a reasonable range
    x_line = np.linspace(0, 12, 300).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color="#F5A623", linewidth=2.2, label="Regression Line", zorder=3)

# Highlight the user's prediction
ax.scatter(
    [study_hours],
    [predicted_score_clamped],
    color="#FF4B4B",
    s=160,
    zorder=5,
    label=f"Your Prediction ({study_hours}h → {predicted_score_clamped:.1f})",
    edgecolors="white",
    linewidths=1.5,
)
ax.axvline(study_hours, color="#FF4B4B", linestyle="--", alpha=0.35, linewidth=1)
ax.axhline(predicted_score_clamped, color="#FF4B4B", linestyle="--", alpha=0.35, linewidth=1)

# Styling
ax.set_xlabel("Hours Studied", color="white", fontsize=11)
ax.set_ylabel("Exam Score", color="white", fontsize=11)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")
ax.legend(
    facecolor="#1a1a2e",
    edgecolor="#333333",
    labelcolor="white",
    fontsize=9,
)
ax.set_ylim(bottom=0)

st.pyplot(fig, use_container_width=True)

# ── Interpretation ────────────────────────────────────────────────────────────
st.divider()
st.subheader("💡 Interpretation")

if predicted_score_clamped >= 75:
    st.success(
        f"A student who studies for **{study_hours:.1f} hours** is predicted to score "
        f"**{predicted_score_clamped:.1f}**. That's a solid result — keep up the effort!"
    )
elif predicted_score_clamped >= 50:
    st.warning(
        f"A student who studies for **{study_hours:.1f} hours** is predicted to score "
        f"**{predicted_score_clamped:.1f}**. Increasing study time could push the score higher."
    )
else:
    st.error(
        f"A student who studies for **{study_hours:.1f} hours** is predicted to score "
        f"**{predicted_score_clamped:.1f}**. Significantly more study time is recommended."
    )

st.caption(
    "ℹ️ This prediction is based on a Linear Regression model trained on the "
    "`students_score.csv` dataset. Results are estimates and individual outcomes may vary."
)
