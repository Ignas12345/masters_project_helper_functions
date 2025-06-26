import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection

#definition of identity transformation, which is used when no transformation is needed.
def identity(df, mode = 'test', **kwargs):
    if mode == 'train':
        train_params = {}
        return df.copy(), train_params
    elif mode == 'test':
        return df.copy()
    else:
        raise ValueError("mode must be either 'train' or 'test'")

#below is the definition of our only sample_wise normalization method - normalization by a housekeeping gene or a list of such genes.

def normalize_by_housekeeping_list(df, housekeeping_list: list, factor = 1, scale_by_housekeep_mean = False, mode = 'test', housekeep_mean = None, **kwargs):
    """
    Written by ChatGPT
    Sample-wise scaling. Normalize miRNA expression df by housekeeping gene(s).
    Assumes:
    - Rows = samples
    - Columns = miRNAs

    Parameters:
    - df: pandas DataFrame (rows = samples, columns = miRNAs), raw counts
    - housekeeping_list: list of miRNA names (column names) to use as reference
    - factor: normalization factor (default = 1)

    Returns:
    - normalized_df: pandas DataFrame of normalized expression values
    """
    # Check that all HK miRNAs exist in columns
    if mode == 'train':
        train_params = {}

    missing = [mir for mir in housekeeping_list if mir not in df.columns]
    if missing:
        raise ValueError(f"Missing housekeeping miRNAs in input df: {missing}")

    # Reference = mean expression of housekeeping miRNA(s) for each sample
    hk_expr = df[housekeeping_list]
    if len(housekeeping_list) > 1:
        reference = hk_expr.mean(axis=1)
    else:
        reference = hk_expr.iloc[:, 0]

    if scale_by_housekeep_mean:
        if mode == 'train':
            # Scale by mean of housekeeping genes; factor is ignored in this case
            factor = reference.mean()
            train_params['housekeep_mean'] = factor
        elif mode == 'test':
            if housekeep_mean is None:
                raise ValueError("housekeep_mean must be provided in test mode when scale_by_housekeep_mean is True")
            factor = housekeep_mean
        else:
            raise ValueError("mode must be either 'train' or 'test'")

    normalized_df = df.div(reference / factor, axis=0)

    if mode == 'train':
        return normalized_df, train_params
    elif mode == 'test':
        return normalized_df

# Below are feature filtering methods, such as filtering by class means, filtering by low expression, and keeping top N features.

def feature_filtering_by_class_means(df, class1_samples = None, class2_samples = None, class1_name = '1', class2_name = '2', filtering_by_class_means_threshold_to_keep = 45, mode = 'test', filtering_by_class_means_features_to_keep = None, **kwargs):
    '''paprasta funkcija, kuri atlieka bruozu filtravima pagal meginius ir ju klases. (should be used with rpm df) - thereshold_to_keep = 50 buvo paimta iš TCGA tyrimo'''
    if mode == 'train':
        train_params = {}
        if class1_samples is None or class2_samples is None:
            raise ValueError("class1_samples and class2_samples must be provided in train mode")
        # Extract expression df for the two classs
        data1 = df.loc[class1_samples].copy()
        data2 = df.loc[class2_samples].copy()

        #By default leave only the features that reach 50 (45 for safety) RPM in any of the classes
        features_to_keep_1 = [feature for feature in data1.columns if data1[feature].mean() >= filtering_by_class_means_threshold_to_keep]
        features_to_keep_2 = [feature for feature in data2.columns if data2[feature].mean() >= filtering_by_class_means_threshold_to_keep]
        filtering_by_class_means_features_to_keep = list(set(features_to_keep_1).union(set(features_to_keep_2)))

        print('number of features expressed (in mean) above ' + str(filtering_by_class_means_threshold_to_keep)+' in class ' +class1_name+ ': ' + str(len(features_to_keep_1)))
        print('number of features expressed (in mean) above ' + str(filtering_by_class_means_threshold_to_keep)+' in class ' +class2_name+ ': ' + str(len(features_to_keep_2)))
        print('number of features in kept across both classes: ' + str(len(filtering_by_class_means_features_to_keep)))
        train_params['filtering_by_class_means_features_to_keep'] = filtering_by_class_means_features_to_keep
        return df[filtering_by_class_means_features_to_keep].copy(), train_params
        
    elif mode == 'test':
        if filtering_by_class_means_features_to_keep is None:
            raise ValueError("filtering_by_class_means_features_to_keep must be provided in test mode")
        print('number of features in kept across both classes for test set: ' + str(len(filtering_by_class_means_features_to_keep)))
        return df[filtering_by_class_means_features_to_keep].copy()
    else:
        raise ValueError("mode must be either 'train' or 'test'")

def filter_low_expression_features_on_raw_counts(df, unsup_filtering_min_count=5, unsup_filtering_min_observations=3, mode = 'test', unsup_filtering_features_to_keep = None, **kwargs):
    """
    Filters out features (columns) that are weakly expressed across observations (should be used with raw counts df).

    Parameters:
    - df (pd.DataFrame): Rows are observations, columns are features (e.g., raw counts).
    - unsup_filtering_min_count (int): Minimum value a feature must have to be considered "expressed".
    - unsup_filtering_min_observations (int): Minimum number of observations where a feature must be expressed.

    Returns:
    - pd.DataFrame: Filtered DataFrame with only sufficiently expressed features.
    """
    if mode == 'train':
        expressed_mask = (df >= unsup_filtering_min_count)
        keep_features = expressed_mask.sum(axis=0) >= unsup_filtering_min_observations
        print('Number of features kept: ' + str(keep_features.sum()))
        unsup_filtering_features_to_keep = keep_features[keep_features].index
        train_params = {'unsup_filtering_features_to_keep': unsup_filtering_features_to_keep}
        return df[unsup_filtering_features_to_keep].copy(), train_params
    elif mode == 'test':
        if unsup_filtering_features_to_keep is None:
            raise ValueError("unsup_filtering_features_to_keep must be provided in test mode")
        print('Number of features kept for test set: ' + str(len(unsup_filtering_features_to_keep)))
        return df[unsup_filtering_features_to_keep].copy()
    else:
        raise ValueError("mode must be either 'train' or 'test'")

def keep_top_n_features_by_mean(df, n, mode = 'test', top_n_columns_by_mean = None, **kwargs):
    if mode == 'train':
        train_params = {}
        # Calculate the mean of each column
        col_means = df.mean()
        # Get the top n columns based on their means
        top_n_columns_by_mean = col_means.nlargest(n).index
        train_params['top_n_columns_by_mean'] = top_n_columns_by_mean
        print('Number of features kept by top_n_features_to_keep: ' + str(len(top_n_columns_by_mean)))
        return df[top_n_columns_by_mean].copy(), train_params
    elif mode == 'test':
        if top_n_columns_by_mean is None:
            raise ValueError("top_n_columns_by_mean must be provided in test mode")
        print('Number of features kept for test set: ' + str(len(top_n_columns_by_mean)))
        return df[top_n_columns_by_mean].copy()

    return top_n_columns_by_mean

def prepare_feature_filtering_methods():
    """
    Prepares a dictionary of feature filtering methods for easy access.
    """
    return {
        'identity': identity,
        'feature_filtering_by_class_means': feature_filtering_by_class_means,
        'filter_low_expression_features_on_raw_counts': filter_low_expression_features_on_raw_counts,
        'keep_top_n_features_by_mean': keep_top_n_features_by_mean
    }

'''Below are feature-wise scaling methods, such as z-normalization, log-transformation. All assume that rows are samples and columns are features.'''

def log_normalization(df, mean_center = True, mode = 'test', means_of_logs = None, **kwargs):
    if mode == 'train':
        train_params = {}
        # Log normalization
        norm_data = np.log2(df + 1)
        if mean_center:
            means_of_logs = norm_data.mean()
            train_params['means_of_logs'] = means_of_logs
            norm_data = norm_data - means_of_logs
        return norm_data, train_params
    elif mode == 'test':
        # Log normalization
        norm_data = np.log2(df + 1)
        if mean_center:
            if means_of_logs is None:
                raise ValueError("means_of_logs must be provided in test mode when mean_center is True")
            norm_data = norm_data - means_of_logs
        return norm_data
    else:
        raise ValueError("mode must be either 'train' or 'test'")

def z_normalization(df, use_std = True, mode = 'test', means = None, stds = None, **kwargs):
    if mode == 'train':
        train_params = {}
        means = df.mean()
        stds = df.std()
        train_params['means'] = means
        train_params['stds'] = stds
        if use_std:
            return (df - means) / (stds + 1e-6), train_params
        else:
            return (df - means), train_params
    elif mode == 'test':
        if means is None:
            raise ValueError("means must be provided in test mode")
        if use_std:
            if stds is None:
                raise ValueError("stds must be provided in test mode if use_std is True")
            return (df - means) / (stds + 1e-6)
        else:
            return (df - means)
    else:
        raise ValueError("mode must be either 'train' or 'test'")

def log_normalization_followed_by_z_normalization(df, log_norm_mean_center = False, z_norm_use_std = True, mode = 'test', train_params_log_norm = None, train_params_z_norm = None, **kwargs):
    if mode == 'train':
        log_norm_data, train_params_log_norm = log_normalization(df, mean_center = log_norm_mean_center, mode = 'train')
        z_norm_on_log_norm_data, train_params_z_norm = z_normalization(log_norm_data, use_std = z_norm_use_std, mode = 'train')
        train_params = {'train_params_log_norm' : train_params_log_norm, 'train_params_z_norm' : train_params_z_norm}
        return z_norm_on_log_norm_data, train_params
    elif mode == 'test':
        if train_params_log_norm is None or train_params_z_norm is None:
            raise ValueError("train_params_log_norm and train_params_z_norm must be provided in test mode")
        log_norm_data = log_normalization(df, mode = 'test', **train_params_log_norm)
        return z_normalization(log_norm_data, use_std = z_norm_use_std, mode = 'test', **train_params_z_norm)

def modified_z_normalization(df, mode = 'test', means = None, stds = None, **kwargs):
    '''
    Same as Z-normalization, but features on bigger scales get more importance
    '''
    if mode == 'train':
        train_params = {}
        means = df.mean()
        stds = df.std() 
        train_params['means'] = means
        train_params['stds'] = stds
        return ((df - means) / (stds + 1e-6)) * np.log(means + 1), train_params
    elif mode == 'test':
        if means is None or stds is None:
            raise ValueError("means and stds must be provided in test mode")
        return ((df - means) / (stds + 1e-6)) * np.log(means + 1)
    else:
        raise ValueError("mode must be either 'train' or 'test'")
         

    
#return z_normalization(log_normalization(df, means = means, means_of_logs = means_of_logs, stds = stds), use_std = True, means = means, stds = stds) * np.log(means + 1)

def prepare_normalization_methods():
    """
    Prepares a dictionary of normalization methods for easy access.
    """
    return {
        'identity': identity,
        'z_normalization': z_normalization,
        'modified_z_normalization': modified_z_normalization,
        'log_normalization': log_normalization,
        'log_normalization_followed_by_z_normalization': log_normalization_followed_by_z_normalization,
    }

#Below are some feature ranking methods and a method for feature filtering with respect to some feature ranking):
def perform_DE_ranking(df_to_use, class1_samples = None, class2_samples = None,  ignore_p_value=False, **kwargs):
        if class1_samples is None or class2_samples is None:
            raise ValueError("For training mode, class1_samples and class2_samples must be provided.")
        # Extract expression data for the two groups
        data1 = df_to_use.loc[class1_samples].copy()
        data2 = df_to_use.loc[class2_samples].copy()

        # Perform unpaired t-test
        p_vals = []
        fold_changes = []
        for feature in df_to_use.columns:
            vals1 = data1[feature].values.astype(float)
            vals2 = data2[feature].values.astype(float)
            t_stat, p = ttest_ind(vals1, vals2, equal_var=False)
            p_vals.append(p)

            fc = np.median(vals1 + 1) / np.median(vals2 + 1)
            if fc < 1:
                fc =  - 1 / fc
            # Median-based fold change
            fold_changes.append(fc)

        # FDR correction
        reject, p_adj = fdrcorrection(p_vals, alpha=0.05)
        print(p_adj)
        results = pd.DataFrame({
            'feature': df_to_use.columns,
            'Fold Change': fold_changes,
            'abs_Fold Change': np.abs(fold_changes),
            'pval': p_vals,
            'fdr': p_adj,
            'significant': reject
        })
        #now sort features firstly significantly, then by absolute fold change
        if ignore_p_value:
            results['significant'] = True
        results = results.sort_values(by=['significant', 'abs_Fold Change'], ascending=[False, False])
        #return only the 'feature' column in order of significance

        return results['feature'].tolist()

def perform_rfe_ranking(df_to_use, class1_samples , class2_samples, estimator_for_rfe_ranking = LogisticRegression(), step_size_for_rfe_ranking = 1, **kwargs):
    y = np.array([1] * len(class1_samples) + [0] * len(class2_samples))
    X = df_to_use.loc[class1_samples + class2_samples].copy()
    # Ensure the estimator is fitted on the combined data
    rfe = RFE(estimator=estimator_for_rfe_ranking, n_features_to_select=1, step=step_size_for_rfe_ranking)
    print('Performing RFE ranking with estimator: ', estimator_for_rfe_ranking.__class__.__name__)
    rfe.fit(X, y)
    # Get the ranking of features
    ranking = rfe.ranking_
    # return the features as a list sorted by their ranking:
    ranked_features = pd.Series(ranking, index=X.columns).sort_values()
    return ranked_features.index.tolist()

    

def rank_features_and_keep_top_n_features(X, class1_samples, class2_samples,  feature_ranking_method, n = 100, drop_correlated_features = False, spearman_corr_threshold = 0.9, mode = 'test', features_to_keep = None, **kwargs):
    """
    Rank features based on a specified feature selection method and keep the top N features.
    
    Parameters:
    - X: DataFrame with rows as samples and columns as features.
    - class1_samples: List of sample indices for class 1.
    - class2_samples: List of sample indices for class 2.
    - n: Number of top features to keep.
    - feature_ranking_method: Function to rank features (e.g., ANOVA, t-test).
    - drop_correlated_features: Whether to drop highly correlated features.
    - spearman_corr_threshold: Correlation threshold for dropping features.
    - mode: 'train' or 'test'.
    - features_to_keep: List of features to keep (required in 'test' mode).
    
    Returns:
    - In 'train' mode: Tuple (DataFrame with top N ranked features, train_params dictionary).
    - In 'test' mode: DataFrame with top N ranked features.
    """
    if mode == 'train':
        train_params = {}
        # Rank features using the specified method
        print('Ranking features using method: ', feature_ranking_method.__name__)
        ranked_features = feature_ranking_method(X, class1_samples, class2_samples, **kwargs)
        
        if drop_correlated_features:
            # Drop highly correlated features
            corr_matrix = X[ranked_features].corr(method='spearman')
            selected = []
            for feat in ranked_features:
                # if feat is correlated above threshold with any already selected, skip it
                if any(corr_matrix.loc[feat, kept] > spearman_corr_threshold for kept in selected):
                    continue
                selected.append(feat)
            ranked_features = selected
        # Keep only the top N features
        if n is None or n > len(ranked_features):
            n = len(ranked_features)
        ranked_features = ranked_features[:n]

        train_params['features_to_keep'] = ranked_features
        return X[ranked_features].copy(), train_params

    elif mode == 'test':
        if features_to_keep is None:
            raise ValueError("features_to_keep must be provided in test mode")
        return X[features_to_keep].copy()
    
    else:
        raise ValueError("mode must be either 'train' or 'test'")