import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sl_utils.logger import pipeline_logger  as logger  # Logger integration
import json

sns.set_style("whitegrid")


def log_dataframe(df, name, level="info"):
    """Helper function to log DataFrames as JSON"""
    log_func = getattr(logger, level)
    log_func(f"{name}:\n{df.head(5).to_json(orient='records', indent=2)}")


def preprocessor_process(source_df, low_cardinality_features=None,
                         skewed_numeric_features=None, numericfeatures=None,
                         high_cardinality_features=None, target=None):
    """Preprocess data and log transformations."""
    logger.info("Starting data preprocessing...")

    transformers = []

    if skewed_numeric_features:
        transformers.append(('numskew',
                             PowerTransformer(method='yeo-johnson',
                                                         standardize=True),
                             skewed_numeric_features))
    if numericfeatures:
        transformers.append(('num',
                             StandardScaler(),
                             numericfeatures))
    if high_cardinality_features:
        transformers.append(('ord',
                             OrdinalEncoder(handle_unknown="use_encoded_value",
                                            unknown_value=-1),
                             high_cardinality_features))
    if low_cardinality_features:
        transformers.append(('cat',
                             OneHotEncoder(handle_unknown='ignore'),
                             low_cardinality_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline([("preprocessor",
                          preprocessor)])
    
    transformed_data = pipeline.fit_transform(source_df)

    if issparse(transformed_data):
        transformed_data = transformed_data.toarray()

    processed_df = pd.DataFrame(transformed_data)
    
    # Assign column names
    all_feature_names = []
    if low_cardinality_features:
        all_feature_names += pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(low_cardinality_features).tolist()
    if skewed_numeric_features:
        all_feature_names += skewed_numeric_features
    if numericfeatures:
        all_feature_names += numericfeatures
    if high_cardinality_features:
        all_feature_names += high_cardinality_features
    
    processed_df.columns = all_feature_names
    processed_df[target] = source_df[target]

    log_dataframe(processed_df, "Processed DataFrame")
    
    logger.info("Preprocessing completed.")
    return processed_df


def check_correlation(processed_df, target):
    """Check correlation and log results."""
    logger.info("Checking feature correlation...")
    correlation = processed_df.corr()[target].sort_values(ascending=False)
    logger.debug(f"Feature Correlations:\n{correlation.to_json(indent=2)}")
    return correlation


def feature_clean_and_selection(processed_df,
                                target,
                                variance_threshold=0.01,
                                correlation_threshold=0.9):
    """Feature selection based on variance and correlation"""
    logger.info("Starting feature selection...")
    
    X = processed_df.drop(columns=[target])
    
    # Remove low-variance features
    var_thresh = VarianceThreshold(threshold=variance_threshold)
    X_reduced = var_thresh.fit_transform(X)
    selected_features = X.columns[var_thresh.get_support()]

    removed_features = list(set(X.columns) - set(selected_features))
    logger.info(f"Removed low-variance features: {removed_features}")

    # Remove highly correlated features
    corr_matrix = X[selected_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                      k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    X_reduced = X[selected_features].drop(columns=to_drop)
    logger.info(f"Removed high-correlation features: {to_drop}")

    return X_reduced, selected_features


def recursive_feature_elimination(X_reduced, y):
    """Recursive Feature Elimination (RFE)"""
    logger.info("Performing Recursive Feature Elimination (RFE)...")

    rfe_selector = RFE(RandomForestClassifier(n_estimators=100,
                                              random_state=42),
                       n_features_to_select=10)
    X_rfe = rfe_selector.fit_transform(X_reduced, y)
    selected_features = X_reduced.columns[rfe_selector.support_]

    logger.info(f"Selected features after RFE: {list(selected_features)}")
    return X_rfe, selected_features


def model_prep(processed_df, target, test_size=0.2, random_state=80):
    """Prepare training and testing data"""
    logger.info("Splitting data into training and testing sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        processed_df.drop([target], axis=1),
        processed_df[target],
        test_size=test_size,
        random_state=random_state
    )

    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def model_training(X_train, y_train, selected_features):
    """Train RandomForest model and log feature importances"""
    logger.info("Training RandomForest model...")

    model = RandomForestClassifier()
    X_train_selected = X_train[selected_features]
    model.fit(X_train_selected, y_train)

    importances = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)
    logger.info(f"Feature Importances:\n{importances.to_json(indent=2)}")

    return model


def model_evaluation(X_train, y_train, X_test, y_test, model):
    """Evaluate model performance and log results"""
    logger.info("Evaluating model performance...")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)

    logger.info(f"Train Classification Report:\n{json.dumps(train_report, indent=2)}")
    logger.info(f"Test Classification Report:\n{json.dumps(test_report, indent=2)}")

    return train_report, test_report


def main():
    """Main pipeline execution"""
    logger.info("Starting main pipeline execution...")

    # Load original settings
    source_df, numericfeatures, skewed_numeric_features, high_cardinality_features, low_cardinality_features, target, target_map = original_settings()

    # Preprocessing
    processed_df = preprocessor_process(source_df, low_cardinality_features, skewed_numeric_features, numericfeatures, high_cardinality_features, target)

    # Feature Selection
    X_reduced, selected_features = feature_clean_and_selection(processed_df, target)

    # RFE
    X_rfe, selected_features = recursive_feature_elimination(X_reduced, processed_df[target])

    # Data Split
    X_train, X_test, y_train, y_test = model_prep(X_rfe, target)

    # Model Training
    model = model_training(X_train, y_train, selected_features)

    # Model Evaluation
    model_evaluation(X_train, y_train, X_test, y_test, model)

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()