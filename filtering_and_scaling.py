from utils import collapse_columns_by_string, filter_by_suffix

def initial_processing_of_TCGA_mirna_counts_pipeline_1(df, samples_to_keep):
    collapsed = collapse_columns_by_string(df)
    filtered = filter_by_suffix(
        collapsed,
        suffix_to_not_include=('unannotated', 'precursor', 'stemloop')
    )
    # Paliekami tik pirmi 15 simbolių indekse
    filtered.index = filtered.index.str[:15]
    filtered = filtered.loc[samples_to_keep].copy()
    print("shape after filtering according to samples" + str(filtered.shape))
    return filtered