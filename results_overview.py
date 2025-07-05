import utils

from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ast

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

def get_aggregated_by_neighbors_weight_matrix(weight_matrix, weight_array, mirna_cluster_df):
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
        
  return aggregated_weight_matrix

def inspect_neighborhoods(features, neighborhood_df, expression_df, weight_array):
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

      corr_df = pd.DataFrame(index = ['avg. weight', 'avg. expression', 'margin', 'corr. with representative'], columns =[feature,])
      corr_df.loc['avg. weight', feature] = weight_array[feature].copy()
      corr_df.loc['avg. expression', feature] = expression_df[feature].mean()
      corr_df.loc['margin', feature] = expression_df[feature].max() - expression_df[feature].min()
      #corr_df.loc['avg. weight diff.', feature] = weight_array[feature] - weight_array[feature]
      corr_df.loc['corr. with representative', feature] = spearmanr(expression_df[feature], expression_df[feature]).correlation
      for nb in neighborhood:
        if nb != feature:
          corr_df.loc['avg. weight', nb] = weight_array.loc[nb].copy()
          corr_df.loc['avg. expression', nb] = expression_df[nb].mean()
          corr_df.loc['margin', nb] = expression_df[nb].max() - expression_df[nb].min()
          #corr_df.loc['avg. weight diff.', nb] = weight_array.loc[nb] - weight_array.loc[feature]
          corr_df.loc['corr. with representative', nb] = spearmanr(expression_df[feature], expression_df[nb]).correlation
      display(corr_df)
      print('\n')

#Functions for creating a graph and plotting the results:

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

def build_feature_graph(A_adj: pd.DataFrame,
                        W: pd.Series,
                        top_k_edges: int = 50,
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