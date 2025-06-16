import pandas as pd

def collapse_columns_by_string(df, string_slice=slice(-1, -12, -1)):
    """
    Collapses columns in a DataFrame based on matching string of column names (last `string_len` characters).

    Args:
        df (pd.DataFrame): DataFrame with features as columns and samples as rows.
        

    Returns:
        pd.DataFrame: DataFrame with collapsed columns (features), summed across matching key_strings.
    """

    print('initial shape: ' + str(df.shape))
    key_strings = df.columns.to_series().str[string_slice]
    # Group column names by their string
    grouped = {}
    for string in key_strings.unique():
        cols = key_strings[key_strings == string].index.tolist()
        grouped[string] = cols

    # Build new DataFrame with collapsed columns
    collapsed_data = {}
    for string, cols in grouped.items():
        # Sum across the columns that match this string
        collapsed_data[cols[0]] = df[cols].sum(axis=1)

    # Construct new DataFrame
    collapsed_df = pd.DataFrame(collapsed_data)
    print('shape after collapsing: ' + str(collapsed_df.shape))

    return collapsed_df

def filter_by_suffix(df, suffix_to_not_include = None, suffix_to_keep = None):
  print('initial shape: ' + str(df.shape))
  if suffix_to_not_include is not None:
    df = df.loc[:, ~df.columns.str.endswith(suffix_to_not_include)]
  if suffix_to_keep is not None:
    df = df.loc[:, df.columns.str.endswith(suffix_to_keep)]
  print('shape after filtering: ' + str(df.shape))
  return df