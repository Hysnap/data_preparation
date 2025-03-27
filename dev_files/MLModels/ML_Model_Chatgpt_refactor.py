import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from itertools import chain
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sns.set_style("whitegrid")


def preprocess_data(source_df, low_cardinality=None, skewed_numeric=None,
                     numeric=None, high_cardinality=None, target=None
                     , run_name="NotSet"):
    """Preprocesses the data by encoding categorical features and scaling numeric features."""
    transformers = []

    if skewed_numeric:
        transformers.append(('num_skew', PowerTransformer(method='yeo-johnson', standardize=True), skewed_numeric))
    if numeric:
        transformers.append(('num', StandardScaler(), numeric))
    if high_cardinality:
        transformers.append(('ord', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), high_cardinality))
    if low_cardinality:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), low_cardinality))

    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline([("preprocessor", preprocessor)])
    transformed_data = pipeline.fit_transform(source_df)

    if issparse(transformed_data):
        transformed_data = transformed_data.toarray()

    processed_df = pd.DataFrame(transformed_data)

    feature_names = []
    if low_cardinality:
        feature_names.extend(
            pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(low_cardinality)
        )
    if skewed_numeric:
        feature_names.extend(skewed_numeric)
    if numeric:
        feature_names.extend(numeric)
    if high_cardinality:
        feature_names.extend(high_cardinality)

    processed_df.columns = feature_names
    processed_df[target] = source_df[target]
    return processed_df


def check_correlation(processed_df, target):
    """Returns correlation of features with the target variable."""
    return processed_df.corr()[target].sort_values(ascending=False)


def train_model(processed_df, target):
    """Trains a RandomForest model and returns feature importances."""
    X = processed_df.drop(columns=[target])
    y = processed_df[target]
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, importances


def feature_selection(processed_df, target, var_threshold=0.01, corr_threshold=0.9):
    """Removes low-variance and highly correlated features."""
    X = processed_df.drop(columns=[target])
    selector = VarianceThreshold(threshold=var_threshold)
    X_reduced = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()]

    corr_matrix = X[selected_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    X_final = X[selected_features].drop(columns=to_drop)

    return X_final, list(X_final.columns)


def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluates model performance using confusion matrix and classification report."""
    results = {}
    for name, X, y in zip(['train', 'test'], [X_train, X_test], [y_train, y_test]):
        y_pred = model.predict(X)
        results[f'{name}_conf_matrix'] = confusion_matrix(y, y_pred)
        results[f'{name}_report'] = classification_report(y, y_pred, output_dict=True)
    return results


def main():
    """Main execution function."""
    source_df = pd.read_csv("data/combined_data.csv")

    numeric = ["month", "day", "year", "week_of_year", "is_weekend", "is_weekday"]
    skewed_numeric = ["title_length", "text_length"]
    high_cardinality = ["location"]
    low_cardinality = ["media_type", "source_name", "subject"]
    target = "label"

    processed_df = preprocess_data(
        source_df, low_cardinality, skewed_numeric, numeric, high_cardinality, target
    )

    correlation = check_correlation(processed_df, target)
    X_reduced, selected_features = feature_selection(processed_df, target)

    X_train, X_test, y_train, y_test = split_data(X_reduced, processed_df[target])
    model, importances = train_model(processed_df, target)
    evaluation_results = evaluate_model(model, X_train, y_train, X_test, y_test)

    outputs = {
        "correlation": correlation,
        "selected_features": selected_features,
        "importances": importances,
        "evaluation": evaluation_results
    }

    return outputs

if __name__ == "__main__":
    results = main()
    print(results)
