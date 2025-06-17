def normalize_by_housekeeping_list(df, housekeeping_mirnas: list, factor = 1):
    """
    Written by ChatGPT
    Normalize miRNA expression data by housekeeping gene(s).
    Assumes:
    - Rows = samples
    - Columns = miRNAs

    Parameters:
    - df: pandas DataFrame (rows = samples, columns = miRNAs), raw counts
    - housekeeping_mirnas: list of miRNA names (column names) to use as reference
    - factor: normalization factor (default = 1)

    Returns:
    - normalized_df: pandas DataFrame of normalized expression values
    """
    # Check that all HK miRNAs exist in columns
    missing = [mir for mir in housekeeping_mirnas if mir not in df.columns]
    if missing:
        raise ValueError(f"Missing housekeeping miRNAs in input data: {missing}")

    # Reference = mean expression of housekeeping miRNA(s) for each sample
    hk_expr = df[housekeeping_mirnas]
    if len(housekeeping_mirnas) > 1:
        reference = hk_expr.mean(axis=1)
    else:
        reference = hk_expr.iloc[:, 0]

    normalized_df = df.div(reference * factor, axis=0)

    return normalized_df