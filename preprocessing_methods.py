import pandas as pd
import numpy as np

#definition of identity transformation, which is used when no transformation is needed.
def identity(df, mode = 'test'):
    if mode == 'train':
        train_params = {}
        return df.copy(), train_params
    elif mode == 'test':
        return df.copy()
    else:
        raise ValueError("mode must be either 'train' or 'test'")

#below is the definition of our only sample_wise normalization method - normalization by a housekeeping gene or a list of such genes.

def normalize_by_housekeeping_list(df, housekeeping_list: list, factor = 1, scale_by_housekeep_mean = False, mode = 'test', housekeep_mean = None):
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

def feature_filtering_by_class_means(df, class1_samples, class2_samples, class1_name = '1', class2_name = '2', filtering_by_class_means_threshold_to_keep = 45, mode = 'test', filtering_by_class_means_features_to_keep = None):
    '''paprasta funkcija, kuri atlieka bruozu filtravima pagal meginius ir ju klases. (should be used with rpm df) - thereshold_to_keep = 50 buvo paimta iš TCGA tyrimo'''
    if mode == 'train':
        train_params = {}
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

def filter_low_expression_features_on_raw_counts(df, unsup_filtering_min_count=5, unsup_filtering_min_observations=3, mode = 'test', unsup_filtering_features_to_keep = None):
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

def keep_top_n_features_by_mean(df, n, mode = 'test', top_n_columns = None):
    if mode == 'train':
        train_params = {}
        # Calculate the mean of each column
        col_means = df.mean()
        # Get the top n columns based on their means
        top_n_columns = col_means.nlargest(n).index
        train_params['top_n_columns'] = top_n_columns
        print('Number of features kept by top_n_features_to_keep: ' + str(len(top_n_columns)))
        return df[top_n_columns].copy(), train_params
    elif mode == 'test':
        if top_n_columns is None:
            raise ValueError("top_n_columns must be provided in test mode")
        print('Number of features kept for test set: ' + str(len(top_n_columns)))
        return df[top_n_columns].copy()

    return top_n_columns

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

def log_normalization(df, mean_center = True, mode = 'test', means_of_logs = None):
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

def z_normalization(df, use_std = True, mode = 'test', means = None, stds = None):
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

def log_normalization_followed_by_z_normalization(df, log_norm_mean_center = False, z_norm_use_std = True, mode = 'test', train_params_log_norm = None, train_params_z_norm = None):
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

def modified_z_normalization(df, mode = 'test', means = None, stds = None):
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