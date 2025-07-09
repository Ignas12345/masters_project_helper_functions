import utils

from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ast
from IPython.display import display

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, brier_score_loss, matthews_corrcoef
import warnings
from sklearn.exceptions import UndefinedMetricWarning

'''
Functions that get frequencies and weights of features selected in each fold:
'''
def get_feature_frequency_matrix(results_df, fold_indices = None, column_name_of_feature_column = 'features_used'):
  if type(fold_indices) == int:
    fold_indices = [fold_indices]
  elif fold_indices is None:
    fold_indices = results_df.index

  results_df = results_df.loc[fold_indices].copy()


  results_df[column_name_of_feature_column] = results_df[column_name_of_feature_column].apply(lambda x: ast.literal_eval(x))
  all_selected_features = sorted(set(f for features in results_df[column_name_of_feature_column] for f in features))

  # Initialize feature_frequency_df with columns
  feature_frequency_df = pd.DataFrame(0, index=results_df.index, columns=all_selected_features)

  for idx, features in results_df[column_name_of_feature_column].items():
        feature_frequency_df.loc[idx, features] = 1

  return feature_frequency_df

def parse_space_separated_list(list_string):
  '''This is a helper function for "get_feature_weight_matrix"'''
  # Remove the brackets and split by spaces
  try:
    numbers_as_strings = list_string.strip('[]').split(',')
    return [float(num) for num in numbers_as_strings]
  except ValueError:
    numbers_as_strings = list_string.strip('[]').split()
    return [float(num) for num in numbers_as_strings]

def get_feature_weight_matrix(results_df, fold_indices = None, column_name_features = 'features_used', column_name_weights = 'feature_importances'):
  if type(fold_indices) == int:
    fold_indices = [fold_indices]
  elif fold_indices is None:
    fold_indices = results_df.index

  results_df = results_df.loc[fold_indices].copy()

  results_df[column_name_features] = results_df[column_name_features].apply(lambda x: ast.literal_eval(x))
  results_df[column_name_weights] = results_df[column_name_weights].apply(parse_space_separated_list)
  all_selected_features = sorted(set(f for features in results_df[column_name_features] for f in features))

  # Initialize selected_feature_df with columns
  selected_feature_weight_df = pd.DataFrame(float(0), index=results_df.index, columns=all_selected_features)


  for idx in results_df.index:
    for j ,feature in enumerate(results_df.loc[idx, column_name_features]):
      selected_feature_weight_df.loc[idx, feature] = results_df.loc[idx, column_name_weights][j]

  return selected_feature_weight_df

def normalize_weight_matrix(weight_matrix, fold_indices = None):
  #normalizes weight matrix row-wise so that each row sums to 1 (in this way the result of each row of the weight matrix should become interpretable)
  if fold_indices is not None:
    weight_matrix = weight_matrix.loc[fold_indices].copy()
  else:
    weight_matrix = weight_matrix.copy()
  weight_matrix = weight_matrix.astype(float)

  weight_matrix = abs(weight_matrix)
  for row in weight_matrix.index:
    normalized_row = weight_matrix.loc[row].divide(weight_matrix.loc[row].sum())
    weight_matrix.loc[row] = normalized_row

  return weight_matrix

def get_average_weight_array(weight_matrix, fold_indices = None, normalize = True):
  #normalization normalizes weight matrix row-wise so that each row sums to 1 (in this way the result of each row of the weight matrix should become interpretable)
  #if normalization is enabled you can think of this as the following: these values either are the avarage weight of the feature across all chosen folds
  #including folds where weight of that feature is 0. Alternatively it is the mean weight of the feature in the fold that it appears in (diagonal value of the adj matrix that would be constructed)
  #from this weight matrix) multiplied by the probability (relative frequency) of that feature appearing in a fold out of the given folds.
  if normalize:
    weight_matrix = normalize_weight_matrix(weight_matrix, fold_indices)

  else:
    if fold_indices is not None:
      weight_matrix = weight_matrix.loc[fold_indices].copy()
    else:
      weight_matrix = weight_matrix.copy()

  weight_array = weight_matrix.sum(axis=0)
  weight_array = weight_array / len(weight_matrix)
  #sort values by absolute weight:
  abs_weight_array = weight_array.abs()
  sorted_indices = abs_weight_array.sort_values(ascending=False).index
  weight_array = weight_array.loc[sorted_indices]
  #make the name of the first column be 'weight'
  weight_array.rename('weight', inplace=True)

  return weight_array

def get_frequency_array(frequency_matrix, fold_indices = None, normalize = False):
  #jeigu be normalizacijos, parodo, kiek kartų per pasirinktas skiltis buvo atrinktas mūsų bruožas.
  #jeigu su normalizacija, tai parodo, tiesiog kokią dalį per visas skiltis atrinktų bruožų sudaro tas bruožas
  if fold_indices is not None:
    frequency_matrix = frequency_matrix.loc[fold_indices].copy()
  else:
    frequency_matrix = frequency_matrix.copy()

  frequency_array = frequency_matrix.sum(axis=0)
  if normalize:
    frequency_array = frequency_array / frequency_array.sum()
    frequency_array.rename('normalized frequency', inplace=True)
  else:
    frequency_array = frequency_array / len(frequency_matrix)
    #have at most 1 as frequency (was used before chaninging how freq_matrix is calculated)
    #frequency_array = frequency_array.clip(upper=1.0)
    frequency_array.rename('frequency', inplace=True)
  return frequency_array.sort_values(ascending=False)

  '''
  Functions for clustering mirnas together based on the miRCarta database:
  '''
def return_neighborhood_mirnas(mirna_name, cluster_df, mirnas_to_use = None):
  if mirnas_to_use is not None:
    mirna_cluster_df = cluster_df.loc[mirnas_to_use].copy()
  else:
    mirna_cluster_df = cluster_df.copy()

  clust_name = mirna_cluster_df.loc[mirna_name, 'Cluster ID']
  if clust_name != clust_name: #this line checks if clust_name is nan
    return [mirna_name,]
  else:
    neighborhood_mirnas=mirna_cluster_df[mirna_cluster_df['Cluster ID'] == clust_name].index
    return neighborhood_mirnas

def get_aggregated_by_neighbors_weight_matrix(weight_matrix, weight_array, mirna_cluster_df, return_freq_matrix = False):
  used_mirnas = []
  weight_matrix = weight_matrix.copy()
  weight_array = weight_array.copy()
  mirnas = weight_array.index

  aggregated_weight_matrix = pd.DataFrame(index=weight_matrix.index)

  for mirna in mirnas:
    if mirna not in used_mirnas:
      neighborhood_mirnas = return_neighborhood_mirnas(mirna, mirna_cluster_df)
      neighborhood_mirnas = [nb for nb in neighborhood_mirnas if nb in mirnas]
      for nb in neighborhood_mirnas:
        used_mirnas.append(nb)

      if len(neighborhood_mirnas) == 1:
        aggregated_weight_matrix[mirna] = weight_matrix[mirna].copy()
        weight_matrix.drop(mirna, axis =1, inplace=True)
        weight_array.drop(mirna, inplace=True)

      else:
        #choose a representative mirna as the mirna from neighborhood mirnas with maximum weight in weight_array
        representative_mirna = abs(weight_array[neighborhood_mirnas]).sort_values(ascending=False).index[0]
        #if len(representative_mirna) != 1:
          #representative_mirna = representative_mirna[0]
        new_row_name = representative_mirna + ' neighborhood'
        # add new feature called new_row_name
        aggregated_weight_matrix[new_row_name] = weight_matrix[representative_mirna].copy()
        weight_matrix.drop(representative_mirna, axis =1, inplace=True)
        weight_array.drop(representative_mirna, inplace=True)
        #now add the rows from neighborhood mirnas to the row of representative mirna:
        for neighborhood_mirna in neighborhood_mirnas:
          if neighborhood_mirna != representative_mirna:
            aggregated_weight_matrix[new_row_name] += weight_matrix[neighborhood_mirna]
            weight_matrix.drop(neighborhood_mirna, axis =1, inplace=True)
  
  if return_freq_matrix:
    #create a frequency matrix for the aggregated weight matrix
    freq_matrix = pd.DataFrame(0, index=aggregated_weight_matrix.index, columns=aggregated_weight_matrix.columns)
    for idx in aggregated_weight_matrix.index:
      for col in aggregated_weight_matrix.columns:
        if aggregated_weight_matrix.loc[idx, col] != 0:
          freq_matrix.loc[idx, col] = 1
    #print(freq_matrix.sum(axis=0))
    return aggregated_weight_matrix, freq_matrix
  else:
    return aggregated_weight_matrix

#Functions for creating a graph and plotting the results:
def construct_adj_matrix_from_freq_matrix(freq_matrix, features_to_use = None):
  #construct an undirected adjacency matrix from the frequency matrix where A[i, j] = number of folds where both i and j were selected / by total number of folds
  if features_to_use is not None:
    freq_matrix = freq_matrix[features_to_use]
  features = freq_matrix.columns
  adj = pd.DataFrame(0.0, index=features, columns=features)
  for i in features:
    for j in features:
      if i != j:
        # Select rows where both feature i and j were selected (non-zero frequency)
        rows_with_i_and_j = freq_matrix[(freq_matrix[i] == 1) & (freq_matrix[j] == 1)]
        # Calculate the proportion of such rows
        adj.loc[i, j] = len(rows_with_i_and_j)
  adj = adj/len(freq_matrix) # Normalize by total number of folds
  return adj

def construct_adj_matrix_from_weight_matrix(weight_matrix, features_to_use = None):
    weight_matrix = normalize_weight_matrix(weight_matrix)
    if features_to_use is not None:
      weight_matrix = weight_matrix[features_to_use]
    #gal čia nebūtina normalizuoti
    #weight_matrix /= len(weight_matrix)
    features = weight_matrix.columns

    adj = pd.DataFrame(0.0, index=features, columns=features)

    for i in features:
      # Select rows where feature i was selected (non-zero weight)
      rows_with_i = weight_matrix[weight_matrix[i] != 0]
      # Sum the weights of all features in those rows
      adj.loc[i, :] = rows_with_i.sum(axis=0) / len(rows_with_i)
    return adj

def build_feature_graph_based_on_freq_adj(A: pd.DataFrame,
                        W: pd.Series,
                        threshold_to_keep_edge: float = 0.1,
                        seed: int = 42):
    


    # Numeric node labels
    id_map = {feat: idx+1 for idx, feat in enumerate(W.index)}
    feats = W.index.tolist()

    G = nx.Graph()
    for feat, node_id in id_map.items():
        G.add_node(node_id, avg_weight=abs(W[feat]))

    # Add edges: only one per pair (i<j)
    for i, feat_i in enumerate(feats):
        for feat_j in feats[i+1:]:
            w_ij = A.loc[feat_i, feat_j]
            if w_ij != 0 and abs(w_ij):
                u = id_map[feat_i]
                v = id_map[feat_j]
                G.add_edge(u, v, weight=w_ij)

    # Compute layout
    
    G_display = nx.Graph()
    G_display.add_nodes_from(G.nodes(data=True))
    top_edges = []
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        if weight >= threshold_to_keep_edge:
            top_edges.append((u, v, weight))

    for u, v, weight in top_edges:
        G_display.add_edge(u, v, weight=weight)

    isolates = list(nx.isolates(G_display))
    if isolates:
        G_display.remove_nodes_from(isolates)

    pos = nx.spring_layout(G, seed=seed)

    # Drawing attributes
    node_sizes = [data['avg_weight'] * 500 for _, data in G_display.nodes(data=True)]
    edge_widths = [data['weight'] * 3 for _, _, data in G_display.edges(data=True)]

    # Plot
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G_display, pos, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G_display, pos, width=edge_widths, alpha=0.7)
    nx.draw_networkx_labels(G_display, pos, font_size=10)

    # Edge labels
    edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in G_display.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G_display, pos,
        edge_labels=edge_labels,
        font_color='black',
        font_size=8,
        label_pos=0.6,
        bbox=dict(facecolor='white', edgecolor='none', pad=0.1)
    )

    plt.title(f"Feature appearance together rates \n Edges with frequency >= {threshold_to_keep_edge}")
    plt.axis('off')
    plt.show()

    return G, id_map

def build_feature_graph_based_on_weight_adj(A_adj: pd.DataFrame,
                        W: pd.Series,
                        top_k_edges: int = 10,
                        seed: int = 42):
    """
    Build and visualize a directed feature influence graph showing only the top K edges.

    Nodes are numbered 1..n in descending order of |W|, and edges are ranked by strength.
    Layout is computed on the full graph, but only the strongest top_k_edges are drawn.

    Parameters
    ----------
    A_adj : pd.DataFrame
        Square adjacency DataFrame (n_features × n_features) where A_adj.loc[i, j]
        measures j's influence on i.
    W : pd.Series
        1D series mapping feature name to weight.
    top_k_edges : int
        Number of strongest edges to display (but all edges influence the layout).
    seed : int
        Random seed for layout reproducibility.

    Returns
    -------
    G_full : networkx.DiGraph
        Full directed graph with all edges.
    G_display : networkx.DiGraph
        Subgraph containing only the top_k_edges strongest edges.
    mapping : dict
        Maps numeric node ID to original feature name.
    """
    # Sort features by descending absolute weight
    sorted_feats = W.abs().sort_values(ascending=False).index.tolist()

    # Align adjacency DataFrame
    A = A_adj.reindex(index=sorted_feats, columns=sorted_feats).fillna(0)

    # Numeric node labels
    id_map = {feat: idx+1 for idx, feat in enumerate(sorted_feats)}

    # Build full graph
    G_full = nx.DiGraph()
    for feat, node_id in id_map.items():
        G_full.add_node(node_id, avg_weight=abs(W[feat]))

    # Collect all edge strengths
    edge_list = []  # (u, v, strength)
    unscaled_stengths = {}
    for i_feat in sorted_feats:
        for j_feat in sorted_feats:
          if i_feat != j_feat:
            w_ij = A.loc[i_feat, j_feat]
            if w_ij != 0:
                u = id_map[j_feat]
                v = id_map[i_feat]
                strength = w_ij * abs(W[i_feat])
                unscaled_stengths[(u, v)] = w_ij
                #strength = abs(w_ij)
                edge_list.append((u, v, strength))

    # Sort edges by strength descending
    edge_list.sort(key=lambda x: x[2], reverse=True)

    # Add all edges to full graph
    for u, v, strength in edge_list:
        G_full.add_edge(u, v, strength=strength)

    # Determine top K edges for display
    top_edges = edge_list[:top_k_edges]
    G_display = nx.DiGraph()
    G_display.add_nodes_from(G_full.nodes(data=True))
    for u, v, strength in top_edges:
        G_display.add_edge(u, v, strength=strength)

    # Compute layout on full graph
    pos = nx.spring_layout(G_full, seed=seed)

    # Prepare drawing attributes
    node_sizes = [data['avg_weight'] * 500 for _, data in G_full.nodes(data=True)]
    edge_widths = [data['strength'] * 5 for _, _, data in G_display.edges(data=True)]

    # Plot
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G_full, pos, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G_display, pos, width=edge_widths, arrowsize=12, alpha=0.7)
    nx.draw_networkx_labels(G_full, pos, font_size=10)

    '''
    edge_strength_labels = {
    (u, v): f"{G_display[u][v]['strength']:.2f}"
    for u, v in G_display.edges()
    }
    '''
    edge_strength_labels = {
    (u, v): f"{unscaled_stengths[(u, v)]:.2f}"
    for u, v in G_display.edges()
    }
    
    nx.draw_networkx_edge_labels(
        G_display, pos,
        edge_labels=edge_strength_labels,
        font_color='black',
        font_size=8,
        label_pos=0.8,          # midpoint
        bbox=dict(               # optional: put label on a white box
            facecolor='white',
            edgecolor='none',
            pad=0.1
        ))

    plt.title(f"Feature Influence Graph (Top {top_k_edges} Edges)")
    plt.axis('off')
    plt.show()

    return G_full, G_display, id_map

'''wrapper functions that display/save the results:'''

def inspect_weight_array(weight_array, threshold = 0.9):
  weight_array = weight_array.copy()
  #plot cumsum of weight_array
  x_values = range(1, len(weight_array) + 1)
  y_values = weight_array.cumsum()
  plt.plot(x_values, y_values)
  plt.xlabel('Number of features')
  plt.ylabel('Cumulative weight')
  idx = np.where(y_values >= threshold)[0][0]
  print('Takes ' + str(idx+1) + f' features/neighborhoods to reach cumsum of {threshold} in weight array')
  #move index to column feature name:
  weight_array = weight_array.reset_index()
  #weight_array.index = weight_array.index + 1
  weight_array.columns = ['feature', 'weight']
  print(weight_array.iloc[:(idx+1)])
  '''
  for i in range(0, 35, 3):
    if i < len(weight_array):
      print(f'cumsum of {i+1} features: ', y_values[i])
  '''
  return (idx+1)

def inspect_neighborhoods(features, neighborhood_df, expression_df, freq_array, weight_array, save_to_latex = True, file_name = None):
  expression_df = expression_df.copy()
  neighborhood_df = neighborhood_df.copy()
  weight_array = weight_array.copy()

  for feature in features:
    #check if feature name ends in 'neighborhood'
    if feature.endswith(' neighborhood'):
      print(f'{feature} consists of : ')
      feature = feature[:-13]
      neighborhood = return_neighborhood_mirnas(feature, neighborhood_df, weight_array.index)
      print(neighborhood)

      corr_df = pd.DataFrame(index = ['frequency', 'avg. weight given app.', 'avg. expression', 'corr. with representative'], columns =[feature,])
      corr_df.loc['frequency', feature] = freq_array[feature].copy()
      try:
        corr_df.loc['avg. weight given app.', feature] = weight_array[feature]/freq_array[feature]
      except ZeroDivisionError:
        corr_df.loc['avg. weight given app.', feature] = 0
      corr_df.loc['avg. expression', feature] = expression_df[feature].mean()
      #corr_df.loc['margin', feature] = expression_df[feature].max() - expression_df[feature].min()
      #corr_df.loc['avg. weight diff.', feature] = weight_array[feature] - weight_array[feature]
      corr_df.loc['corr. with representative', feature] = spearmanr(expression_df[feature], expression_df[feature]).correlation

      for nb in neighborhood:
        if nb != feature:

          corr_df.loc['frequency', nb] = freq_array[nb].copy()
          try:
            corr_df.loc['avg. weight given app.', nb] = weight_array[nb]/freq_array[nb]
          except ZeroDivisionError:
            corr_df.loc['avg. weight given app.', nb] = 0
          #corr_df.loc['avg. weight', nb] = weight_array.loc[nb].copy()
          corr_df.loc['avg. expression', nb] = expression_df[nb].mean()
          #corr_df.loc['margin', nb] = expression_df[nb].max() - expression_df[nb].min()
          #corr_df.loc['avg. weight diff.', nb] = weight_array.loc[nb] - weight_array.loc[feature]
          corr_df.loc['corr. with representative', nb] = spearmanr(expression_df[feature], expression_df[nb]).correlation
      display(corr_df)
      if save_to_latex:
        if file_name is None:
          raise ValueError('file_name must be provided if save_to_latex is True')
        corr_df.to_latex(f'{file_name}_{feature}_neighborhood_table.tex', float_format= "%.2f")
      print('\n')

def display_results(result_df, fold_indices = None, mirna_cluster_df = None, use_aggregated_results = False, inspect_agg_neighborhoods = False, expression_df = 'None', save_to_latex = True, save_to_csv = True, file_name = None, threshold_to_keep_edge = None, top_k_edges_for_graph = None):

  if save_to_latex or save_to_csv:
    if file_name is None:
      raise ValueError('file_name must be provided if save_to_latex or save_to_csv is True')

  if fold_indices is not None:
    result_df = result_df.loc[fold_indices].copy()
  else:
    result_df = result_df.copy()
  freq_matrix =  get_feature_frequency_matrix(result_df)
  freq_array = get_frequency_array(freq_matrix, normalize=False)
  weight_matrix = get_feature_weight_matrix(result_df)
  weight_array = get_average_weight_array(weight_matrix, normalize=True)
  unnormalized_weight_array = get_average_weight_array(weight_matrix, normalize=False)

  if use_aggregated_results:
    if mirna_cluster_df is None:
      raise ValueError('mirna_cluster_df must be provided if use_aggregated_results is True')
    orig_weight_array = weight_array.copy()
    orig_freq_array = freq_array.copy()
    weight_matrix, freq_matrix = get_aggregated_by_neighbors_weight_matrix(weight_matrix, weight_array, mirna_cluster_df, return_freq_matrix=True)

    freq_array = get_frequency_array(freq_matrix, normalize=False)
    weight_array = get_average_weight_array(weight_matrix, normalize=True)
    unnormalized_weight_array = get_average_weight_array(weight_matrix, normalize=False)


  k = inspect_weight_array(weight_array, threshold = 0.9)
  top_k_features = weight_array.iloc[:k].index
  cumsum_of_top_k_features = weight_array[top_k_features].sum()
  print(f'cumsum of top {k} features: ' + str(cumsum_of_top_k_features))

  print('Frequency array: ')
  display(freq_array.iloc[:])

  print('Normalized weight array: ')
  display(weight_array.iloc[:k])

  print('Unnormalized weight array: ')
  display(unnormalized_weight_array.iloc[:k])
  '''
  #A_adj = construct_adj_matrix_from_weight_matrix(weight_matrix, features_to_use=top_k_features)
  W = weight_array[top_k_features]

  if top_k_edges_for_graph is None:
    top_k_edges = 10
  else:
    top_k_edges = top_k_edges_for_graph
  build_feature_graph_based_on_weight_adj(A_adj, W, top_k_edges=top_k_edges)
  '''
  A_adj = construct_adj_matrix_from_freq_matrix(freq_matrix)
  if threshold_to_keep_edge is None:
    threshold_to_keep_edge = 0.1
  W = weight_array.copy()
  build_feature_graph_based_on_freq_adj(A_adj, W, threshold_to_keep_edge=threshold_to_keep_edge)
  #convert W to df with one column 'avg. normalized weight':
  final_feat_df = pd.DataFrame(index = top_k_features)
  final_feat_df['frequency'] = freq_array[top_k_features]
  final_feat_df['avg. norm. weight given appearance'] = weight_array[top_k_features]/freq_array[top_k_features]
  final_feat_df['avg. weight given appearance'] = unnormalized_weight_array[top_k_features]/freq_array[top_k_features]
  final_feat_df = final_feat_df.reset_index()
  final_feat_df.index +=1
  display(final_feat_df)
  print('\n')
  print("Adjacency matrix for graph: ")
  display(A_adj)

  if save_to_latex:
    freq_array.to_latex(f'{file_name}_freq_array_table.tex', float_format= "%.2f")
    weight_array.to_latex(f'{file_name}_weight_array_table.tex', float_format= "%.2f")
    unnormalized_weight_array.to_latex(f'{file_name}_unnormalized_weight_array_table.tex', float_format= "%.2f")
    final_feat_df.to_latex(f'{file_name}_final_features_table.tex', float_format= "%.2f")
  if save_to_csv:
    freq_array.to_csv(f'{file_name}_freq_array.csv', sep = ';')
    weight_array.to_csv(f'{file_name}_weight_array.csv', sep = ';')
    unnormalized_weight_array.to_csv(f'{file_name}_unnormalized_weight_array.csv', sep = ';')
    final_feat_df.to_csv(f'{file_name}_final_features.csv', sep = ';')

  if inspect_agg_neighborhoods and use_aggregated_results:
    if expression_df is None:
      raise ValueError('expression_df must be provided if inspect_agg_neighborhoods is True')
    inspect_neighborhoods(top_k_features, mirna_cluster_df, expression_df, freq_array=orig_freq_array, weight_array = orig_weight_array, save_to_latex=save_to_latex, file_name=file_name)

metrics_dict = {
    'Accuracy' : accuracy_score,
    'Balanced accuracy' : balanced_accuracy_score,
    'Recall' : recall_score,
    'Precision' : precision_score,
    'MCC' : matthews_corrcoef,
    'Brier loss' : brier_score_loss
}

def calculate_metrics_for_loocv_folds(results_df, fold_indices, metrics_dict):
  results_df = results_df.loc[fold_indices].copy()
  metrics_df = pd.DataFrame(index = results_df.index, columns = metrics_dict.keys())
  y_test = []
  y_proba = []
  y_pred = []
  n_features = []
  for row in results_df.index:
    y_test.append(parse_space_separated_list(results_df.loc[row, 'true label']))
    y_proba.append(parse_space_separated_list(results_df.loc[row, 'prob_class_1']))
    y_pred.append(parse_space_separated_list(results_df.loc[row, 'prediction']))
    n_features.append(results_df.loc[row, 'number_of_features_used'])

  for metric in metrics_dict.keys():
      if metric == 'brier_score_loss':
        metrics_df.loc[:, metric] = metrics_dict[metric](y_test, y_proba)
      else:
        metrics_df.loc[:, metric] = metrics_dict[metric](y_test, y_pred)
  metrics_df['Feature number'] = n_features
        
  return metrics_df

#define Undefined warning as an error:
warnings.filterwarnings("error", category=UndefinedMetricWarning)

def calculate_metrics_for_non_loocv_folds(results_df, fold_indices, metrics_dict):
  if fold_indices is not None:
    results_df = results_df.loc[fold_indices].copy()
  else:
    results_df = results_df.copy()
  metrics_df = pd.DataFrame(index = results_df.index, columns = metrics_dict.keys())
  for row in results_df.index:
    y_test = parse_space_separated_list(results_df.loc[row, 'true label'])
    y_proba = parse_space_separated_list(results_df.loc[row, 'prob_class_1'])
    y_pred = parse_space_separated_list(results_df.loc[row, 'prediction'])
    n_features = results_df.loc[row, 'number_of_features_used']

    try:
      for metric in metrics_dict.keys():
        if metric == 'brier_score_loss':
          metrics_df.loc[row, metric] = metrics_dict[metric](y_test, y_proba)
        else:
          metrics_df.loc[row, metric] = metrics_dict[metric](y_test, y_pred)
    except Exception as e:
      print(f"Error in row {row}: {e}")
      continue
    metrics_df.loc[row, 'Feature number'] = n_features

  return metrics_df

def get_fold_metrics(results_df, train_folds_df, metrics_dict = metrics_dict, percentile_for_conf_int = 0.95, save_to_latex = True, file_name = None):
  alpha = 1 - percentile_for_conf_int

  # We'll build a list of Series (one per comment), then concat them into one DataFrame
  summary_series = []

  for comment in train_folds_df['comment'].unique():
      if comment == 'all training samples':
          continue
      print(comment)
      # get your fold indices and metrics
      fold_indices = utils.get_folds_by_comment(train_folds_df, comment)
      if comment == 'loocv fold':
          metrics_df = calculate_metrics_for_loocv_folds(results_df, fold_indices=fold_indices, metrics_dict=metrics_dict)
      else:
          metrics_df = calculate_metrics_for_non_loocv_folds(results_df, fold_indices=fold_indices, metrics_dict=metrics_dict)

      # compute mean and CI bounds
      mean_vals  = metrics_df.mean()
      lower_vals = metrics_df.quantile(alpha/2)
      upper_vals = metrics_df.quantile(1 - alpha/2)

      # format each metric as "mean; CI: [lower, upper]"
      if comment == 'loocv fold':
        formatted = mean_vals.index.to_series().apply(
          lambda metric: (
             f"{(100*mean_vals[metric]):.1f}"
                if (metric != 'Brier loss' and metric != 'Feature number')
                else f"{(mean_vals[metric]):.3f}"
                ))
      else:
        formatted = mean_vals.index.to_series().apply(
            lambda metric: (
                 f"{(100*mean_vals[metric]):.1f} CI:[{100*lower_vals[metric]:.1f}, {100*upper_vals[metric]:.1f}]"
                if (metric != 'Brier loss' and metric != 'Feature number')
                else f"{(mean_vals[metric]):.3f} CI:[{lower_vals[metric]:.3f}, {upper_vals[metric]:.3f}]"
                ))

      # name this Series by the comment, so concat will make it a column
      formatted.name = comment + 's'
      summary_series.append(formatted)

  # put them all side by side
  summary_df = pd.concat(summary_series, axis=1)

  # now summary_df.index are your metric names, columns are each comment
  display(summary_df)
  if save_to_latex:
    if file_name is None:
      raise ValueError('file_name must be provided if save_to_latex is True')
    summary_df.to_latex(f'{file_name}_metrics_summary_table_with_{percentile_for_conf_int}_conf_ints.tex')
  return summary_df