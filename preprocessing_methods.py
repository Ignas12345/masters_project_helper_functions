def normalize_by_housekeeping_list(df, housekeeping_list: list, factor = 1):
    """
    Written by ChatGPT
    Sample-wise scaling. Normalize miRNA expression data by housekeeping gene(s).
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
    missing = [mir for mir in housekeeping_list if mir not in df.columns]
    if missing:
        raise ValueError(f"Missing housekeeping miRNAs in input data: {missing}")

    # Reference = mean expression of housekeeping miRNA(s) for each sample
    hk_expr = df[housekeeping_list]
    if len(housekeeping_list) > 1:
        reference = hk_expr.mean(axis=1)
    else:
        reference = hk_expr.iloc[:, 0]

    normalized_df = df.div(reference * factor, axis=0)

    return normalized_df

def feature_filtering_by_class_means(df_to_use, class1_samples, class2_samples, class1_name = '1', class2_name = '2', threshold_to_keep = 45):
    '''paprasta funkcija, kuri atlieka bruozu filtravima pagal meginius ir ju klases. (should be used with rpm data) - thereshold_to_keep = 50 buvo paimta iš TCGA tyrimo'''
    # Extract expression data for the two classs
    data1 = df_to_use.loc[class1_samples].copy()
    data2 = df_to_use.loc[class2_samples].copy()

    #By default leave only the features that reach 50 (45 for safety) RPM in any of the classes
    features_to_keep_1 = [feature for feature in data1.columns if data1[feature].mean() >= threshold_to_keep]
    features_to_keep_2 = [feature for feature in data2.columns if data2[feature].mean() >= threshold_to_keep]
    features_to_keep = list(set(features_to_keep_1).union(set(features_to_keep_2)))

    print('number of features expressed (in mean) above ' + str(threshold_to_keep)+' in class ' +class1_name+ ': ' + str(len(features_to_keep_1)))
    print('number of features expressed (in mean) above ' + str(threshold_to_keep)+' in class ' +class2_name+ ': ' + str(len(features_to_keep_2)))
    print('number of features in kept across both classes: ' + str(len(features_to_keep)))

    return features_to_keep

def filter_low_expression_features_on_raw_counts(data, min_count=5, min_observations=3):
    """
    Filters out features (columns) that are weakly expressed across observations (should be used with raw counts data).

    Parameters:
    - data (pd.DataFrame): Rows are observations, columns are features (e.g., raw counts).
    - min_count (int): Minimum value a feature must have to be considered "expressed".
    - min_observations (int): Minimum number of observations where a feature must be expressed.

    Returns:
    - pd.DataFrame: Filtered DataFrame with only sufficiently expressed features.
    """
    data = data.copy()
    expressed_mask = (data >= min_count)
    keep_features = expressed_mask.sum(axis=0) >= min_observations
    print('Number of features kept: ' + str(keep_features.sum()))
    return keep_features[keep_features].index

def keep_top_n_features_by_mean(df, n):

    col_means = df.mean()
    top_n_columns = col_means.nlargest(n).index

    return top_n_columns