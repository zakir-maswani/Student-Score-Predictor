# <center> **📚 Student Scores Prediction — Based on Study Hours** </center>

> A machine learning project that uses **Linear Regression** to predict student exam scores from the number of hours studied. Includes a full data preprocessing pipeline, model training/evaluation, and an interactive **Streamlit** web app for real-time predictions.

---

## 📁 Project Structure

```
├── data_preprocessing_and_model_training.ipynb   # Jupyter notebook (EDA + training)
├── students_score.csv                             # Dataset
├── model.pkl                                      # Saved trained model
├── students_streamlit_app.py                                         # Streamlit prediction app
├── requirements.txt                               # Python dependencies
└── README.md
```

---

## 📊 Dataset

The dataset (`students_score.csv`) contains two columns:

| Column   | Description                          |
|----------|--------------------------------------|
| `Hours`  | Number of hours a student studied    |
| `Scores` | Exam score achieved by the student   |

---

## 🔍 Exploratory Data Analysis

The notebook covers:

- **Shape & Info** — dimensions, data types, memory usage
- **Statistical Summary** — mean, std, min/max via `describe()`
- **Null Value Check** — confirms data completeness
- **Distribution Analysis** — horizontal boxplots for both `Hours` and `Scores`

---

## 🤖 Model

**Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)

**Pipeline:**
1. Load and inspect the dataset
2. Split features (`Hours`) and target (`Scores`)
3. Train/test split — 80% train / 20% test (`random_state=42`)
4. Fit the model on training data
5. Predict on both train and test sets
6. Evaluate with R², MSE, and MAE
7. Serialize the trained model to `model.pkl` via `pickle`

---

## 📈 Evaluation Metrics

| Metric                 | Description                              |
|------------------------|------------------------------------------|
| R² Score               | Proportion of variance explained         |
| Mean Squared Error     | Average squared prediction error         |
| Mean Absolute Error    | Average absolute prediction error        |

The regression plot shows predicted vs. actual values alongside the ideal fit line.

---

## 🚀 Running the Streamlit App

### 1. Clone / Download the project

```bash
git clone https://github.com/zakir-maswani/Student-Scores-Prediction.git
cd student-scores-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (if `model.pkl` doesn't exist)

Run the notebook end-to-end, or execute the training script directly. This produces `model.pkl`.

### 4. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ App Features

- Input study hours via an intuitive slider or number input
- Instant score prediction using the saved model
- Visual regression line chart with the input point highlighted
- Model performance metrics displayed in the sidebar

---

## 🛠️ Requirements

See `requirements.txt`. Key dependencies:

- `pandas` — data manipulation
- `numpy` — numerical operations
- `scikit-learn` — model training and evaluation
- `matplotlib` / `seaborn` — visualizations
- `streamlit` — interactive web app

---

## 📌 Key Findings

- Study hours show a **strong positive linear relationship** with exam scores
- Linear Regression achieves a high R² on this dataset, confirming the model fits well
- The feature importance DataFrame confirms `Hours` as the sole predictor coefficient

