# d_transform/ML_model2.py
import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.dropna(subset=["cleaned_text", "label"], inplace=True)  # Ensure required columns are present
    # rename cleaned_text to text
    df.rename(columns={"cleaned_text": "text"}, inplace=True)
    return df


def adjust_realness_score(df: pd.DataFrame) -> pd.DataFrame:
    df["realness_score"] = df["label"].map({0: 1, 1: 5})
    return df


def create_vectorizer(text_series: pd.Series) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    vectorizer.fit(text_series)
    return vectorizer


def vectorize_text(vectorizer: TfidfVectorizer, text_series: pd.Series):
    return vectorizer.transform(text_series)


def train_model(X, y, model_type: str):
    if model_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test, model_type: str) -> dict:
    y_pred = model.predict(X_test)

    if model_type == "classification":
        return {
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Classification Report": classification_report(y_test, y_pred, output_dict=True)
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        return {"Mean Squared Error": mse,
                "Predictions": y_pred.tolist()}


def save_artifacts(model,
                   vectorizer,
                   model_type: str,
                   output_dir: str = "sl_data_for_dashboard"):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/ML_model_{model_type}.pkl")
    joblib.dump(vectorizer, f"{output_dir}/vectorizer_{model_type}.pkl")
    print(f"[✓] Saved model and vectorizer for '{model_type}' in {output_dir}")


def run_pipeline(data_path: str, model_type: str = "regression"):
    assert model_type in ("classification",
                          "regression"), "Invalid model type."

    print(f"▶ Running ML pipeline in '{model_type}' mode...")

    df = load_data(data_path)

    target_column = "label" if model_type == "classification" else "realness_score"
    if model_type == "regression":
        df = adjust_realness_score(df)

    vectorizer = create_vectorizer(df["text"])
    X = vectorize_text(vectorizer, df["text"])
    y = df[target_column]

    (X_train,
     X_test,
     y_train,
     y_test) = train_test_split(X,
                                y,
                                test_size=0.2,
                                random_state=42)

    model = train_model(X_train, y_train, model_type=model_type)
    evaluation = evaluate_model(model, X_test, y_test, model_type=model_type)
    save_evaluation_reports(evaluation, model_type, output_dir="ML_model2_models")
    save_artifacts(model, vectorizer, model_type=model_type, output_dir="ML_model2_models")


def save_evaluation_reports(evaluation, model_type: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    if model_type == "classification":
        # Save classification report as JSON
        with open(f"{output_dir}/classification_report.json", "w") as f:
            json.dump(evaluation["Classification Report"], f, indent=2)

        # Extract summary metrics
        report = evaluation["Classification Report"]
        summary = {
            "accuracy": report.get("accuracy", None),
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1-score": report["weighted avg"]["f1-score"]
        }
        pd.DataFrame([summary]).to_csv(f"{output_dir}/evaluation_summary.csv", index=False)

        # 🆕 Optional: Flatten and save confusion matrix
        confusion = evaluation["Confusion Matrix"]
        flat_conf = {
            "True Negatives": confusion[0][0],
            "False Positives": confusion[0][1],
            "False Negatives": confusion[1][0],
            "True Positives": confusion[1][1]
        }
        pd.DataFrame([flat_conf]).to_csv(f"{output_dir}/confusion_matrix.csv", index=False)

    else:  # regression
        summary = {
            "mean_squared_error": evaluation["Mean Squared Error"]
        }
        pd.DataFrame([summary]).to_csv(f"{output_dir}/evaluation_summary.csv", index=False)

        # For regression, we can save all evaluation results
        evaluation_df = pd.DataFrame([evaluation])
        evaluation_df.to_csv(f"{output_dir}/evaluation_results.csv", index=False)

    # Save explanation file
    explanation = (
        "### Evaluation Metrics Explanation\n"
        "- **Accuracy**: Proportion of correct predictions over total.\n"
        "- **Precision**: How many selected items are relevant.\n"
        "- **Recall**: How many relevant items are selected.\n"
        "- **F1-Score**: Harmonic mean of precision and recall.\n"
        "- **MSE** (regression only): Average squared difference between predicted and actual values.\n"
    )
    with open(f"{output_dir}/evaluation_explanation.txt", "w") as f:
        f.write(explanation)

    print("[✓] Evaluation reports saved (CSV + JSON + TXT).")



if __name__ == "__main__":
    # 👇 You can toggle between "regression" or "classification" here
    # run_pipeline(data_path="sl_data_for_dashboard/ML_train_data.csv",
    #              model_type="regression")
    # or use classification
    run_pipeline(data_path="sl_data_for_dashboard/training_data.zip",
                 model_type="classification")


# PATH: d_transform/ML_model2.py
# end of d_transform/ML_model2.py