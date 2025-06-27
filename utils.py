import pandas as pd

def get_folds_by_comment(train_fold_df, comments: list | str):
  if type(comments) == str:
    comments = [comments]
  indices = []
  for comment in comments:
    indices.extend(train_fold_df[train_fold_df['comment'] == comment].index.tolist())
  return indices

def map_items_to_names(items_dict):
  name_dict = {}
  for key in items_dict:
      try:
        name_dict[key] = items_dict[key].__name__
      except AttributeError:
        name_dict[key] = items_dict[key]
  return name_dict

def collapse_columns_by_string(df, string_slice=slice(-1, -12, -1)):
    """
    Collapses columns in a DataFrame based on matching string of column names (last `string_len` characters).

    Args:
        df (pd.DataFrame): DataFrame with features as columns and samples as rows.
        

    Returns:
        pd.DataFrame: DataFrame with collapsed columns (features), summed across matching key_strings.
    """
    df = df.copy()
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
  df = df.copy()
  print('initial shape: ' + str(df.shape))
  if suffix_to_not_include is not None:
    df = df.loc[:, ~df.columns.str.endswith(suffix_to_not_include)]
  if suffix_to_keep is not None:
    df = df.loc[:, df.columns.str.endswith(suffix_to_keep)]
  print('shape after filtering: ' + str(df.shape))
  return df

def initial_pre_processing_pipeline(df):
    collapsed = collapse_columns_by_string(df)
    filtered = filter_by_suffix(
        collapsed,
        suffix_to_not_include=('unannotated', 'precursor', 'stemloop')
    )
    return filtered

def format_data_frame(df:str|pd.DataFrame, sep = ',', decimal = '.', index_col = 0, columns_slice = slice(15), transpose = False, fill_nan_with_0 = True):
  '''
  Function to format a DataFrame.
  Parameters:
  - df: DataFrame or URL to a CSV file
  - sep: separator for CSV file
  - decimal: decimal separator for CSV file
  - index_col: column to use as index
  - columns_slice: slice to apply to column names
  - transpose: whether to transpose the DataFrame
  - fill_nan_with_0: whether to fill NaN values with 0
  Returns:
  - formatted DataFrame
  '''
  message = ''

  if type(df) == str:
    print('reading df from url: ' + df)
    df = pd.read_csv(df, sep = sep, decimal = decimal, index_col = index_col)
  else:
    df = df.copy()

  #check if there are nan values
  if df.isna().any().any() and fill_nan_with_0:
      df.fillna(0, inplace = True)
      
      print('nan values filled with 0')


  if columns_slice is not None:
    df.columns = [col[columns_slice] for col in df.head().columns]
    print('column names truncated using: ' + str(columns_slice))


  if transpose:
    df = df.T.copy()
    print('df transposed')

  print('final shape of df: ' + str(df.shape) + '\n')

  return df