    """
    description: This script is used to generate outputs for the
    machine learning model
    1. It preprocesses the data
    2. Checks correlation
    3. Visualises features
    4. Tests and trains the model
    5. Cleans and selects features
    6. Implements Recursive Feature Elimination
    7. Plots the model
    8. Plots feature importance
    9. Prepares the model
    10. Evaluates the model
    11. Tests the model
    12. Saves the outputs to a file
    13. Returns the outputs

    Describe model used: RandomForestClassifier
    Describe dataset used:  combined_data.csv
    Describe features used: skewed_numeric_features, numericfeatures,
                            low_cardinality_features, high_cardinality_features
    Describe target used:   label
    Describe target_map used: Fake, Real
    Describe test_size used: 0.2
    Describe random_state used: 80
    Describe variance_threshold used: 0.01
    Describe correlation_threshold used: 0.9
    Describe n_features_to_select used: 10
    """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from sklearn.preprocessing import (StandardScaler,
                                   OneHotEncoder,
                                   OrdinalEncoder,
                                   PowerTransformer)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             )
from c_data_extract_combine.ETL import checkdirectory
sns.set_style("whitegrid")


checkdirectory()

def preprocessor_process(
    source_df,
    low_cardinality_features=None,
    skewed_numeric_features=None,
    numericfeatures=None,
    high_cardinality_features=None,
    target=None,
        ):
    # Define preprocessor
    features = (low_cardinality_features +
                skewed_numeric_features +
                numericfeatures +
                high_cardinality_features)
    # Define preprocessor
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

    # Create pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor)
    ])

    # Fit and transform data
    transformed_data = pipeline.fit_transform(source_df)

    # Convert sparse matrix to DataFrame if necessary
    if issparse(transformed_data):
        transformed_data = transformed_data.toarray()

    processed_df = pd.DataFrame(transformed_data)

    # Numeric features remain the same
    if numericfeatures:
        numeric_feature_names = numericfeatures
    # Ordinal encoded features remain but are numerical
    if high_cardinality_features:
        ordinal_feature_names = high_cardinality_features
    # Get the names for OneHotEncoded features
    if low_cardinality_features:
        onehot_feature_names = (
            list(pipeline.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .get_feature_names_out(low_cardinality_features))
        )
    # Combine them all
    all_feature_names = (list(onehot_feature_names) +
                         list(skewed_numeric_features) +
                         numeric_feature_names +
                         ordinal_feature_names)
    # Assign them to the transformed DataFrame
    processed_df.columns = all_feature_names
    # Merge back the target column
    # Ensure correct alignment!
    processed_df[target] = source_df[target]
    return processed_df


def check_correlation(processed_df):
    # Check correlation - the closer to 1 or -1, the stronger the correlation
    correlation = processed_df.corr()["label"].sort_values(ascending=False)
    return correlation


def feature_visualisations(processed_df, target):
    # Define the features to plot
    all_feature_names = [col for col in processed_df.columns if col != target]
    features_to_plot = all_feature_names

    # Calculate the number of rows needed
    num_features = len(features_to_plot)
    num_cols = 4
    num_rows = (num_features + num_cols - 1) // num_cols

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each feature
    for i, feature in enumerate(features_to_plot):
        sns.boxplot(x=processed_df["label"],
                    y=processed_df[feature],
                    ax=axes[i])
        axes[i].set_title(f"{feature.replace('_', ' ').title()} vs. Label")

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return plt.show()


def test_and_train(processed_df, target):
    # Train a simple model
    X = processed_df.drop(columns=[target])
    y = processed_df["label"]
    model = RandomForestClassifier()
    model.fit(X, y)

    # Get feature importances
    importances = (
        pd.Series(model.feature_importances_,
                  index=X.columns)
        .sort_values(ascending=False))
    return importances


def feature_clean_and_selection(processed_df,
                                target,
                                variance_threshold=0.01,
                                correlation_threshold=0.9):
    # remove low-variance features
    X = processed_df.drop(columns=[target])
    # Remove features with very low variance
    # (e.g., all values are almost the same)
    # Remove near-constant features
    var_thresh = VarianceThreshold(threshold=variance_threshold)
    X_reduced = var_thresh.fit_transform(X)

    # Get the remaining feature names
    selected_features = X.columns[var_thresh.get_support()]
    removed_features = list(set(X.columns)
                            - set(selected_features))
    # produce list of removed features
    removed_features_text = ("Removed Features:",
                             list(removed_features))
    remaining_features_text = ("Remaining Features:",
                               list(selected_features))

    # Compute correlation matrix
    corr_matrix = X[selected_features].corr().abs()

    # Find features with correlation above 0.9
    upper = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                  k=1).astype(bool)))
    to_drop = [
        column for column in upper.columns
        if any(upper[column] > correlation_threshold)
    ]

    # Drop highly correlated features
    X_reduced = X[selected_features].drop(columns=to_drop)
    # produce list of remaining features
    selected_features_post_high_correlation = list(X_reduced.columns)
    dropped_features_post_high_correlation = list(to_drop)
    # produce list of removed features
    removed_features_post_correlation_text = (
        "Removed Features after correlation filtering:", list(to_drop)
        )
    features_post_correlation_text = ("Features after correlation filtering:",
                                      list(X_reduced.columns))
    return (removed_features_text, remaining_features_text,
            removed_features_post_correlation_text,
            features_post_correlation_text,
            selected_features_post_high_correlation,
            dropped_features_post_high_correlation,
            X_reduced)


def recursive_feature_elimination(X_reduced, y):
    # implement Recursive Feature Elimination (RFE) - to train the model
    # and remove least important features
    # Use RandomForest to evaluate feature importance
    rfe_selector = RFE(RandomForestClassifier(n_estimators=100,
                                              random_state=42),
                       n_features_to_select=10)
    X_rfe = rfe_selector.fit_transform(X_reduced, y)

    # Get selected feature names
    selected_features = X_reduced.columns[rfe_selector.support_]
    final_selected_features_text = ("Final Selected Features:",
                                    list(selected_features))
    return final_selected_features_text, selected_features, X_rfe


def Plot(X_reduced, y, selected_features):
    # Train a final model with selected features
    model = RandomForestClassifier()
    model.fit(X_reduced[selected_features], y)


def PlotFeatureImportance(model, selected_features):
    # Plot feature importance
    importances = pd.Series(model.feature_importances_,
                            index=selected_features
                            ).sort_values(ascending=False)
    importances.plot(kind="bar", figsize=(10, 5), title="Feature Importance")
    return plt.show(), importances


def model_prep(processed_df, target, test_size=0.2, random_state=80):
    (X_train,
     X_test,
     y_train,
     y_test) = train_test_split(processed_df.drop(columns=[target]),
                                processed_df[target],
                                test_size=test_size,
                                random_state=random_state)

    dataset_info = {
        "train_set_shape": X_train.shape,
        "train_target_shape": y_train.shape,
        "test_set_shape": X_test.shape,
        "test_target_shape": y_test.shape
    }
    return X_train, X_test, y_train, y_test, dataset_info


def confusion_matrix_and_report(X_train, y_train, pipeline, target_map):
    """
    Gets features, target, pipeline and how labelled (named)
    the levels from your target

    - it predicts based on features
    - compare predictions and actuals in a confusion matrix
        - the first argument stays as rows and the second stay as columns
        in the matrix
        - we will use the pattern where the predictions are in the row and
        actual values are in the columns
        - to refresh that, revert to the Performance Metric video in Module 2
    - show classification report

    """
    prediction = pipeline.predict(X_train)

    results = {}
    results['confusion_matrix'] = (
        pd.DataFrame(confusion_matrix(y_true=prediction, y_pred=y_train),
                     columns=[["Actual " + sub for sub in target_map]],
                     index=[["Prediction " + sub for sub in target_map]]))
    results['classification_report'] = classification_report(
        y_train,
        prediction,
        target_names=target_map,
        output_dict=True)
    return results


def clf_performance(X_train,
                    y_train,
                    X_test,
                    y_test,
                    pipeline,
                    label_map):
    """
    gets the features and target from train and test set, pipeline how
    you labelled (named) the levels from your target
    - for each set (train and test), it calls the function above
    to show the confusion matrix
    and classification report for both train and test set
    """

    train_results = confusion_matrix_and_report(X_train,
                                                y_train,
                                                pipeline,
                                                label_map)

    test_results = confusion_matrix_and_report(X_test,
                                               y_test,
                                               pipeline,
                                               label_map)

    return {"train_results": train_results, "test_results": test_results}


def model_test(X_train,
               y_train,
               X_test,
               y_test,
               model,
               target_map):
    # Train a simple model
    X_live = X_test.iloc[0:50]
    y_live = y_test.iloc[0:50]
    y_pred_live = model.predict(X_live)
    y_pred_prob_live = model.predict_proba(X_live)

    clf_performance(X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    pipeline=model,
                    label_map=[target_map[0], target_map[1]]
                    )
    return y_pred_live, y_pred_prob_live


def original_settings():
    # Load dataset
    source_df = pd.read_csv("data//combined_data.csv")

    # Define features

    skewed_numeric_features = ["title_length", "text_length"]
    numericfeatures = [
        col for col in source_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if col not in skewed_numeric_features
    ]
    low_cardinality_features = ["media_type",
                                "source_name",
                                "subject",
                                "day_label",
                                "day_of_week"
                                ]
    high_cardinality_features = [
        col for col in source_df.select_dtypes(include=['object']).columns
        if col not in low_cardinality_features
    ]
    remaining_features = [
        col for col in source_df.columns
        if col not in high_cardinality_features
        and col not in low_cardinality_features
        and col not in numericfeatures
        and col not in skewed_numeric_features
    ]
    target = "label"
    target_map = ["Fake", "Real"]

    return (source_df,
            numericfeatures,
            skewed_numeric_features,
            high_cardinality_features,
            low_cardinality_features,
            remaining_features,
            target,
            target_map)


def main():
    generateoutputs = False
    # Load original settings
    (source_df,
     numericfeatures,
     skewed_numeric_features,
     high_cardinality_features,
     low_cardinality_features,
     remaining_features,
     target,
     target_map) = original_settings()
    # create a function called run_outputs that will be used to generate
    # outputs all the returned variable from original settings into
    # it as dictionary
    run_outputs = {}
    run_outputs = {"original_settings":
                   {"source_df": source_df.head(5),
                    "numericfeatures": numericfeatures,
                    "skewed_numeric_features": skewed_numeric_features,
                    "high_cardinality_features": high_cardinality_features,
                    "low_cardinality_features": low_cardinality_features,
                    "remaining_features": remaining_features,
                    "target": target,
                    "target_map": target_map
                    }}

    # Preprocess the data
    processed_df = preprocessor_process(source_df,
                                        low_cardinality_features,
                                        skewed_numeric_features,
                                        numericfeatures,
                                        high_cardinality_features,
                                        target)
    # append processed_df.head(5) to run_outputs
    run_outputs["preprocessor_process"] = {"processed_df":
                                           processed_df.head(5)}

    # Check correlation
    correlation = check_correlation(processed_df)

    run_outputs["check_correlation"] = {"correlation": correlation}

    # Feature selection and cleaning
    (removed_features_text, remaining_features_text,
            removed_features_post_correlation_text,
            features_post_correlation_text,
            selected_features_post_high_correlation,
            dropped_features_post_high_correlation,
            X_reduced) = feature_clean_and_selection(processed_df,
                                              target)

    run_outputs["feature_clean"] = {
        "removed_features":   removed_features_text,
        "remaining_features": remaining_features_text,
        "removed_features_post_correlation":
        removed_features_post_correlation_text,
        "features_post_correlation":
        features_post_correlation_text,
        "selected_features_post_high_correlation":
        selected_features_post_high_correlation,
        "dropped_features_post_high_correlation":
        dropped_features_post_high_correlation,
        "X_reduced": X_reduced.head(5)}

    # Recursive feature elimination
    y = processed_df[target]

    (final_selected_features_text,
     selected_features,
     X_rfe) = recursive_feature_elimination(X_reduced, y)

    run_outputs["recursive_feature_elimination"] = {
        "final_selected_features": final_selected_features_text,
        "selected_features": selected_features,
        "X_rfe": X_rfe[:5]}

    # create temp variable to store all selected features and target
    final_selected_features_inc_target = final_selected_features_text[1] + [target]
    # Split the data into train and test sets
    (X_train,
     X_test,
     y_train,
     y_test,
     dataset_info) = model_prep(processed_df[final_selected_features_inc_target],
                                target)

    run_outputs["model_prep"] = {"X_train": X_train.head(5),
                                 "X_test": X_test.head(5),
                                 "y_train": y_train.head(5),
                                 "y_test": y_test.head(5),
                                 "dataset_info": dataset_info}

    # Train the model
    model = RandomForestClassifier()
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    model.fit(X_train_selected, y_train)

    # Plot feature importance
    run_outputs["FeatureImportance"] = {
        "FIPlot": PlotFeatureImportance(model, selected_features)}

    # Evaluate model performance
    performance = clf_performance(X_train_selected,
                                  y_train,
                                  X_test_selected,
                                  y_test,
                                  model,
                                  target_map)
    run_outputs["results"] = {
        "train_performance": performance["train_results"],
        "test_performance": performance["test_results"]
    }
    return run_outputs


if __name__ == "__main__":
    returned = main()
    print(returned)
    print("Done")
    # save returned to a file
    if returned:
        with open("output.txt", "w") as f:
            f.write(str(returned))
            f.close()
    else:
        print("No output returned")
