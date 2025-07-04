import pandas as pd
from copy import deepcopy
import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SequentialFeatureSelector, RFECV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection

def select_all_features(X_train, y_train, **kwargs):
  return X_train.columns

def helper_ffs_calculate_mean_training_score(X, y, classifier, skf):
    """
    Helper function for forward_feature_selection_cv that calculates the mean training score for the given features and labels using a given instance of StratifiedKFold.

    Parameters:
        X (pd.DataFrame): The feature set.
        y (pd.Series or array-like): The labels.
        classifier: The scikit-learn classifier to be used.
        skf (StratifiedKFold): The StratifiedKFold object for splitting.

    Returns:
        float: The mean training score across all folds.
    """
    training_scores = []
    for train_idx, _ in skf.split(X, y):
        classifier.fit(X.iloc[train_idx], np.array(y)[train_idx])
        training_scores.append(classifier.score(X.iloc[train_idx], np.array(y)[train_idx]))
    return np.mean(training_scores)

# Updated function (mostly written by ChatGPT)
def forward_feature_selection_cv(X_cv, y_cv, classifier, max_features=10, allow_n_rounds_without_improvement=5, n_splits=5, seed=42, **kwargs):
    """
    Perform forward feature selection over the provided set of candidate features
    and evaluate the model using stratified k-fold cross-validation. The function
    returns the set of features that yielded the best validation performance.

    Parameters:
        X_cv (pd.DataFrame): The dataframe containing the candidate features (columns).
        y_cv (pd.Series or array-like): The labels corresponding to the rows in X_cv.
        classifier: The scikit-learn classifier to be used.
        max_features (int): Maximum number of features to include. The default is 20.
        allow_n_rounds_without_improvement (int): Number of rounds without improvement before stopping.
        n_splits (int): Number of folds for StratifiedKFold.
        seed (int): Random state seed for reproducibility.

    Returns:
        best_feature_set (list): List of feature names that produced the best CV score.
    """
    # Ensure the classifier is cloned to avoid modifying the original instance
    classifier = clone(classifier)
    candidate_features = list(X_cv.columns)
    current_features = []
    remaining_features = deepcopy(candidate_features)

    best_cv_score_overall = 0
    best_training_score_overall = 0
    best_feature_set = []
    performance_history = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print("Starting forward feature selection with cross-validation...")
    print("Using estimator: ", classifier.__class__.__name__)

    rounds_without_improvement = 0

    while remaining_features and (len(current_features) < max_features):
        if rounds_without_improvement >= allow_n_rounds_without_improvement:
            print("Stopping due to lack of improvement.")
            break

        best_cv_score_this_round = 0
        best_training_score_this_round = 0
        best_feature_this_round = None

        for feature in remaining_features:
            temp_features = current_features + [feature]
            X_sub = X_cv[temp_features]
            cv_scores = cross_val_score(classifier, X_sub, y_cv, cv=skf)
            mean_cv_score = cv_scores.mean()

            if mean_cv_score > best_cv_score_this_round:
                best_cv_score_this_round = mean_cv_score
                best_feature_this_round = feature

            if mean_cv_score == best_cv_score_this_round and best_feature_this_round is not None:
                if best_training_score_this_round == 0.0:
                    X_sub_2 = X_cv[current_features + [best_feature_this_round]]
                    best_training_score_this_round = helper_ffs_calculate_mean_training_score(X_sub_2, y_cv, classifier, skf)

                mean_training_score = helper_ffs_calculate_mean_training_score(X_sub, y_cv, classifier, skf)

                if mean_training_score > best_training_score_this_round:
                    best_feature_this_round = feature
                    best_training_score_this_round = mean_training_score

        if best_feature_this_round is None:
            print("Adding any feature results in cv score of 0.0, stopping.")
            break

        current_features.append(best_feature_this_round)
        remaining_features.remove(best_feature_this_round)
        performance_history.append({
            "num_features": len(current_features),
            "cv_score": best_cv_score_this_round,
            "training_score": best_training_score_this_round,
            "features": deepcopy(current_features)
        })
        print(f"Round {len(current_features)}: Added '{best_feature_this_round}' -> CV Score: {best_cv_score_this_round:.4f}, Training Score: {best_training_score_this_round:.4f}")

        if (best_cv_score_this_round > best_cv_score_overall) or (best_cv_score_this_round == best_cv_score_overall and best_training_score_this_round > best_training_score_overall):
            best_cv_score_overall = best_cv_score_this_round
            best_training_score_overall = best_training_score_this_round
            best_feature_set = deepcopy(current_features)
            rounds_without_improvement = 0
            if best_cv_score_this_round == 1 and best_training_score_this_round == 1:
                print("No improvement possible on the training data performance")
                break
        else:
            rounds_without_improvement += 1

    print("Forward feature selection completed!")
    print(f"Best feature set ({len(best_feature_set)} features) with CV score: {best_cv_score_overall:.4f} and training score: {best_training_score_overall:.4f}")
    #print(f"Best feature set: {best_feature_set}")

    return best_feature_set
    
#ChatGPT funkcija
def forward_feature_selection_cv_without_evaluation_on_train_set(X_cv, y_cv, classifier, max_features=10, allow_n_rounds_without_improvement = 3, n_splits=5, seed=42, **kwargs):
    """
    Perform forward feature selection over the provided set of candidate features
    (which could be the top DE features) and evaluate the model using stratified k-fold
    cross-validation. The function returns the set of features that yielded the best
    validation performance.

    Parameters:
        X_cv (pd.DataFrame): The dataframe containing the candidate features (columns).
        y_cv (array-like): The labels corresponding to the rows in X_cv.
        classifier: The scikit-learn classifier to be used.
        max_features (int): Maximum number of features to include. The default is 20.
        n_splits (int): Number of folds for StratifiedKFold.
        seed (int): Random state seed for reproducibility.

    Returns:
        best_feature_set (list): List of feature names that produced the best CV score.
        performance_history (list of dicts): A log of performance at each round.
    """
    # Clone the classifier to avoid modifying the original instance.
    classifier = clone(classifier)
    # Ensure the candidate features are in a list.
    candidate_features = list(X_cv.columns)
    current_features = []
    remaining_features = deepcopy(candidate_features)

    best_score_overall = 0
    best_feature_set = []
    performance_history = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print("Starting forward feature selection with cross-validation...")
    print("Using estimator: ", classifier.__class__.__name__)

    # Loop until we've reached the maximum allowed features or no more remain
    while remaining_features and (len(current_features) < max_features):
        best_score_this_round = 0
        best_feature_this_round = None

        # Evaluate each candidate feature by adding it to the current set.
        for feature in remaining_features:
            temp_features = current_features + [feature]
            X_sub = X_cv[temp_features]
            # Evaluate with cross-validation.
            scores = cross_val_score(classifier, X_sub, y_cv, cv=skf)
            mean_score = scores.mean()

            if mean_score > best_score_this_round:
                best_score_this_round = mean_score
                best_feature_this_round = feature

        if best_feature_this_round is None:
            # No improvement is possible.
            print('best_feature_this_round is None')
            break

        # Update the current set and remove the selected feature from remaining features.
        current_features.append(best_feature_this_round)
        remaining_features.remove(best_feature_this_round)
        performance_history.append({
            "num_features": len(current_features),
            "cv_score": best_score_this_round,
            "features": deepcopy(current_features)
        })
        print(f"Round {len(current_features)}: Added '{best_feature_this_round}' -> CV Score: {best_score_this_round:.4f}")

        # Update overall best if this round's score improved.
        if best_score_this_round > best_score_overall:
            best_score_overall = best_score_this_round
            best_feature_set = deepcopy(current_features)

        #add stopping criterion: if accuraccy does not improve in n rounds, stop process:
        #if len(performance_history) >= 3 and performance_history[-1]["cv_score"] <= performance_history[-2]["cv_score"] and performance_history[-1]["cv_score"] <= performance_history[-3]["cv_score"]:
        if len(performance_history) >= (allow_n_rounds_without_improvement+1) and performance_history[-1]["cv_score"] <= performance_history[-(allow_n_rounds_without_improvement+1)]["cv_score"]:
            break

    print("Forward feature selection completed!")
    print(f"Best feature set ({len(best_feature_set)} features) with CV score: {best_score_overall:.4f}")
    return best_feature_set

def sfs_feature_selection(X_cv, y_cv, classifier, max_features=10, tol = None, n_splits=5, seed=42, **kwargs):
    """
    Perform forward feature selection using scikit-learn's SequentialFeatureSelector.

    Parameters:
        X_cv (pd.DataFrame): DataFrame of candidate features.
        y_cv (array-like): Target class labels.
        classifier: A scikit-learn compatible classifier.
        n_features (int): Number of features to select (upper limit).
        n_splits (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.

    Returns:
        selected_features (list): List of feature names selected by SFS.
        cv_score (float): Mean cross-validation score with the selected features.
    """
    # Ensure the classifier is cloned to avoid modifying the original instance
    classifier = clone(classifier)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Create the SequentialFeatureSelector object; note that direction='forward' configures forward selection.
    sfs = SequentialFeatureSelector(
        classifier,
        n_features_to_select=max_features,
        tol = tol,
        direction='forward',
        cv=cv
    )

    # Fit the SFS model to perform feature selection.
    sfs.fit(X_cv, y_cv)

    # Get a boolean mask of the selected features.
    selected_mask = sfs.get_support()
    selected_features = X_cv.columns[selected_mask].tolist()

    # Evaluate the performance using cross-validation on the selected features.
    scores = cross_val_score(classifier, X_cv[selected_features], y_cv, cv=cv)
    cv_score = scores.mean()

    print(f"Selected features ({len(selected_features)}): {selected_features}")
    print(f"Cross-Validation Score: {cv_score:.4f}")

    return selected_features

def rfecv_feature_selection(X_cv, y_cv, classifier, max_features=10, n_splits=5, seed=42, **kwargs):
    """
    Perform recursive feature elimination (RFE) to select features. Use sklearn's RFECV

    Parameters:
        X_cv (pd.DataFrame): DataFrame of candidate features.
        y_cv (array-like): Target class labels.
        classifier: A scikit-learn compatible classifier.
        max_features (int): Number of top features to keep.
        n_splits (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.

    Returns:
        selected_features (list): List of feature names selected by RFE.
    """
    # Ensure the classifier is cloned to avoid modifying the original instance
    classifier = clone(classifier)
    print("Running RFE with CV feature selection...")
    print("Using estimator: ", classifier.__class__.__name__)
    rfecv = RFECV(
        estimator=classifier,
        step=1,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        scoring='balanced_accuracy',
    )
    rfecv.fit(X_cv, y_cv)
    selected_features = X_cv.columns[rfecv.support_].tolist()
    if len(selected_features) > max_features:
        selected_features = selected_features[:max_features]
    print(f"Selected features with rfecv ({len(selected_features)}): {selected_features}")
    return selected_features


def keep_top_n_features(X_cv, y_cv, classifier, max_features=10, n_splits=5, seed=42, **kwargs):
    "keep top n features"
    return X_cv.columns[:max_features].tolist()

def keep_top_n_features_by_cv(X_cv, y_cv, classifier, max_features=10, n_splits=5, seed=42, **kwargs):
    """
    Keep the top n features based on cross-validation performance.

    Parameters:
        X_cv (pd.DataFrame): DataFrame of candidate features.
        y_cv (array-like): Target class labels.
        classifier: A scikit-learn compatible classifier.
        max_features (int): Number of top features to keep.
        n_splits (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.

    Returns:
        selected_features (list): List of feature names selected by CV.
    """
    # Ensure the classifier is cloned to avoid modifying the original instance
    classifier = clone(classifier)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Evaluate each feature individually and store their scores
    feature_scores = []
    feature_set = []
    for i,feature in enumerate(X_cv.columns):
        if i> max_features:
            break
        feature_set.append(feature)
        # Use cross-validation to evaluate the performance of the classifier with this feature
        scores = cross_val_score(classifier, X_cv[feature_set], y_cv, cv=cv)
        feature_scores.append(scores.mean())

    best_number_of_features = np.argmax(feature_scores) + 1  # +1 because argmax returns index starting from 0
    selected_features = X_cv.columns[:best_number_of_features].tolist() 

    print(f"Selected top {len(selected_features)} features based on CV performance: {selected_features}")

    return selected_features
