import time

import sklearn.base
from sklearn._config import set_config
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, brier_score_loss, matthews_corrcoef
from sklearn.decomposition import PCA
import ast

import utils
import preprocessing_methods as pp
import plotting

set_config(transform_output='pandas')

def run_pre_processing_on_train_set(X_train, pre_processing_methods, **kwargs):
  print('Running pre processing on train set')
  method_training_estimates_dict = {}
  for method_name in pre_processing_methods:
    method = pre_processing_methods[method_name]
    print('Running ' + method_name + ': ' + str(method.__name__))
    X_train, method_training_estimates = method(X_train, mode = 'train', **kwargs)
    method_training_estimates_dict[method_name] = method_training_estimates

  return X_train, method_training_estimates_dict

def run_pre_processing_on_test_set(X_test, pre_processing_methods, method_training_estimates_dict, **kwargs):
  print('Running pre processing on test set')
  for method_name in pre_processing_methods:
    method = pre_processing_methods[method_name]
    print('Running ' + method_name + ': ' + str(method.__name__))
    method_training_estimates = method_training_estimates_dict[method_name]
    X_test = method(X_test, mode = 'test', **method_training_estimates, **kwargs)

  return X_test

def run_pipeline_for_single_fold(X_train, y_train, X_test, y_test, pre_processing_methods, feature_selection_method, classification_model, **kwargs):
  classification_model = sklearn.base.clone(classification_model)

  #training phase:
  X_train, method_training_estimates_dict = run_pre_processing_on_train_set(X_train, pre_processing_methods = pre_processing_methods, **kwargs)

  print('Feature selection method: ' + str(feature_selection_method.__name__))
  selected_features = feature_selection_method(X_train.copy(), y_train, classifier = kwargs['classifier_for_feature_selection'], **kwargs)
  print(f'Number of selected features: {len(selected_features)}')
  X_train = X_train[selected_features].copy()
  classification_model.fit(X_train, y_train)

  #testing phase:
  X_test = run_pre_processing_on_test_set(X_test, pre_processing_methods = pre_processing_methods, method_training_estimates_dict = method_training_estimates_dict, **kwargs)
  X_test = X_test[selected_features].copy()
  predictions = classification_model.predict(X_test)

  try:
    prob_1 = classification_model.predict_proba(X_test)[:,1]
  except Exception as e:
    print(e)
    print('proba attribute is not defined, calculating decision function')
    dec = classification_model.decision_function(X_test)
    full_dec = classification_model.decision_function(pd.concat([X_train, X_test]))
    prob_1 = (dec - full_dec.min() + 1e-7) / (full_dec.max() - full_dec.min() + 1e-5)

  #get number of features used
  try:
    feature_importances = classification_model.coef_[0]
  except:
    try:
      feature_importances = classification_model.feature_importances_
    except:
      feature_importances = np.ones(len(selected_features))
  number_of_features_used = np.sum(abs(feature_importances) > 0)


  results_dict = {
    'sample' : X_test.index.to_list(),
    'prediction' : predictions,
    'prob_class_1' : prob_1,
    'true label' : y_test,
    'number_of_features_used' : number_of_features_used
  }

  if number_of_features_used <= 20:
    #save non-zero importance features
    mask = np.abs(feature_importances) > 0
    results_dict['features_used'] = [f for f, m in zip(selected_features, mask) if m]
    results_dict['feature_importances'] = [float(round(imp, 5)) for imp, m in zip(feature_importances, mask) if m]
  print('results: ')
  print(results_dict)



  return results_dict

def run_pipeline_across_folds(df, sample_label_dict, train_folds_df, test_folds_df, pre_processing_methods, feature_selection_method, classification_model, fold_indices = None, autosave = True, save_path = '', **kwargs):

  results = []
  if fold_indices is None:
    fold_indices = range(1, len(train_folds_df))
  elif isinstance(fold_indices, int):
    fold_indices = range(fold_indices, len(train_folds_df))

  for fold_index in fold_indices:
    print(f'Fold {fold_index}')

    training_samples = ast.literal_eval(train_folds_df.iloc[fold_index]['samples'])
    testing_samples = ast.literal_eval(test_folds_df.iloc[fold_index]['samples'])
    X_train = df.loc[training_samples].copy()
    y_train = [sample_label_dict[sample] for sample in training_samples]
    y_train = [int(1) if label == kwargs['smaller_group_label'] else int(0) for label in y_train]
    kwargs['class1_samples'] = [sample for sample in X_train.index if sample_label_dict[sample] == kwargs['smaller_group_label']]
    kwargs['n_splits'] = min(len(kwargs['class1_samples']), 5)
    kwargs['class2_samples'] = [sample for sample in X_train.index if sample_label_dict[sample] == kwargs['larger_group_label']]

    X_test = df.loc[testing_samples].copy()
    y_test = [sample_label_dict[sample] for sample in testing_samples]
    y_test = [int(1) if label == kwargs['smaller_group_label'] else int(0) for label in y_test]

    results_for_single_fold = run_pipeline_for_single_fold(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                            pre_processing_methods=pre_processing_methods, feature_selection_method=feature_selection_method,
                                                            classification_model=classification_model, **kwargs)
    results.append(results_for_single_fold)
    results_df  = pd.DataFrame(results)
    
    if autosave:
      results_df.to_csv(save_path + 'results_df_autosave.csv', sep = ';')

  return results_df

def prepare_and_run_pipeline_on_folds(df, fold_path, experiment_name, sample_label_dict, pre_processing_methods, feature_selection_method, classification_method, fold_indices = None, existing_results_df = None, **kwargs):
  
  df = df.copy()
  train_folds_df = pd.read_csv(fold_path + 'train_folds_' + f'{experiment_name}.csv', index_col = 0)
  test_folds_df = pd.read_csv(fold_path + 'test_folds_' + f'{experiment_name}.csv', index_col = 0)
  group_labels = [label for label in list(set(sample_label_dict.values())) if label != 'unused']
  kwargs['smaller_group_label'] = min(group_labels, key=lambda x: len([sample for sample, label in sample_label_dict.items() if label == x]))
  kwargs['larger_group_label'] = max(group_labels, key=lambda x: len([sample for sample, label in sample_label_dict.items() if label == x]))
  print('kwargs passed:' + str(kwargs))

  if 'classification_method_parameters' in kwargs:
      classification_method_parameters = kwargs['classification_method_parameters']
      classification_model = classification_method(**classification_method_parameters)
  else:
      classification_model = classification_method()

  if 'feature_ranking_method' in kwargs and 'estimator_for_rfe_ranking' not in kwargs:
    if kwargs['feature_ranking_method'].__name__ == 'perform_RFE_ranking':
      kwargs['estimator_for_rfe_ranking'] = sklearn.base.clone(classification_model)

  if 'classifier_for_feature_selection' not in kwargs:
    kwargs['classifier_for_feature_selection'] = sklearn.base.clone(classification_model)

  start_time = time.time()
  results_df = run_pipeline_across_folds(df, sample_label_dict=sample_label_dict, train_folds_df=train_folds_df, test_folds_df=test_folds_df, fold_indices=fold_indices, pre_processing_methods=pre_processing_methods,
                                         feature_selection_method=feature_selection_method, classification_model=classification_model, **kwargs)
  end_time = time.time()
  print(f'Pipeline finished in {end_time - start_time} seconds')

  if existing_results_df is not None:
    results_df = pd.concat([existing_results_df, results_df], axis=0)
    results_df = results_df[~results_df.index.duplicated(keep='last')]

  return results_df



def run_pipeline_on_loocv_folds(df, sample_label_dict, train_folds_df, test_folds_df, info_df, loocv_fold_indices,  pre_processing_methods, feature_selection_method, classification_model, **kwargs):

  loocv_results_df = pd.DataFrame(index = info_df.index, columns=['prediction', 'prob_class_1', 'true label'])

  for fold_index in loocv_fold_indices:
    print(f'Fold {fold_index}')

    training_samples = ast.literal_eval(train_folds_df.iloc[fold_index]['samples'])
    testing_sample = ast.literal_eval(test_folds_df.iloc[fold_index]['samples'])
    X_train = df.loc[training_samples].copy()
    y_train = [sample_label_dict[sample] for sample in training_samples]
    y_train = [int(1) if label == kwargs['smaller_group_label'] else int(0) for label in y_train]
    kwargs['class1_samples'] = [sample for sample in X_train.index if sample_label_dict[sample] == kwargs['smaller_group_label']]
    kwargs['n_splits'] = min(len(kwargs['class1_samples']), 5)
    kwargs['class2_samples'] = [sample for sample in X_train.index if sample_label_dict[sample] == kwargs['larger_group_label']]

    X_test = df.loc[testing_sample].copy()
    y_test = [sample_label_dict[sample] for sample in testing_sample]
    y_test = [int(1) if label == kwargs['smaller_group_label'] else int(0) for label in y_test]

    results_for_single_fold = run_pipeline_for_single_fold(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                            pre_processing_methods=pre_processing_methods, feature_selection_method=feature_selection_method,
                                                            classification_model=classification_model, **kwargs)
    for result in results_for_single_fold:
      try:
        loocv_results_df.loc[testing_sample, result] = results_for_single_fold[result]
      except Exception as e:
        loocv_results_df.loc[testing_sample, result] = str(results_for_single_fold[result])

  return loocv_results_df

metrics_dict = {
    'accuracy_score' : accuracy_score,
    'balanced_accuracy_score' : balanced_accuracy_score,
    'recall_score' : recall_score,
    'precision_score' : precision_score,
    'brier_score_loss' : brier_score_loss,
    'matthews_corrcoef' : matthews_corrcoef
}

def calculate_loocv_metrics(loocv_results_df, metrics_dict = metrics_dict, **used_method_names_dict,):

  loocv_metrics_row = pd.Series(data = used_method_names_dict)


  for metric_name in metrics_dict:
    metric = metrics_dict[metric_name]

    if metric_name == 'brier_score_loss':
      loocv_metrics_row[metric_name] = metric(loocv_results_df['true label'].to_list(), loocv_results_df['prob_class_1'].to_list())
    else:
      loocv_metrics_row[metric_name] = metric(loocv_results_df['true label'].to_list(), loocv_results_df['prediction'].to_list())
    if 'number_of_features_used' in loocv_results_df.columns:
      loocv_metrics_row['mean_feature_number'] = np.mean(loocv_results_df['number_of_features_used'].to_list())

  return loocv_metrics_row.to_frame().T

def perform_pca(df, sample_label_dict, pre_processing_methods = None, label_dict_for_plotting = None, use_only_training_samples_for_pca = True, title = '', **kwargs):

  if use_only_training_samples_for_pca:
    samples_for_pca = [sample for sample in sample_label_dict if sample_label_dict[sample] != 'unused']
  else:
    samples_for_pca = df.index

  X_train = df.loc[samples_for_pca].copy()

  if pre_processing_methods is not None:
    if 'smaller_group_label' in kwargs and 'larger_group_label' in kwargs:
      smaller_group_label = kwargs['smaller_group_label']
      larger_group_label = kwargs['larger_group_label']
      smaller_group_training_samples = [sample for sample in X_train.index if sample_label_dict[sample] == smaller_group_label]
      larger_group_training_samples = [sample for sample in X_train.index if sample_label_dict[sample] == larger_group_label]
      kwargs['class1_samples'] = smaller_group_training_samples
      kwargs['class2_samples'] = larger_group_training_samples
      if title is None:
        title = f'{smaller_group_label} vs {larger_group_label}'
    X_train,_ = run_pre_processing_on_train_set(X_train, pre_processing_methods, **kwargs)

  pca = PCA(n_components=2)
  fitted_data = pca.fit_transform(X_train)
  if label_dict_for_plotting is None:
    label_dict_for_plotting = sample_label_dict

  final_labels_for_plotting = {sample : label_dict_for_plotting[sample] for sample in X_train.index}


  plotting.plot_two_features(fitted_data, 'pca0', 'pca1', sample_label_dict = final_labels_for_plotting, show_legend=False, title = title)

def run_pipeline_on_loocv_folds_and_record_performance(df, fold_path, experiment_name, sample_label_dict, pre_processing_methods, feature_selection_method, classification_method, plot_pca = True, **kwargs):

  df = df.copy()
  info_df = pd.read_csv(fold_path + 'info_df_' + f'{experiment_name}.csv', index_col = 0)
  train_folds_df = pd.read_csv(fold_path + 'train_folds_' + f'{experiment_name}.csv', index_col = 0)
  test_folds_df = pd.read_csv(fold_path + 'test_folds_' + f'{experiment_name}.csv', index_col = 0)
  group_labels = [label for label in list(set(sample_label_dict.values())) if label != 'unused']
  kwargs['smaller_group_label'] = min(group_labels, key=lambda x: len([sample for sample, label in sample_label_dict.items() if label == x]))
  kwargs['larger_group_label'] = max(group_labels, key=lambda x: len([sample for sample, label in sample_label_dict.items() if label == x]))
  print('kwargs passed:' + str(kwargs))

  loocv_fold_indices = utils.get_folds_by_comment(train_folds_df, 'loocv fold')

  if 'classification_method_parameters' in kwargs:
      classification_method_parameters = kwargs['classification_method_parameters']
      classification_model = classification_method(**classification_method_parameters)
  else:
      classification_model = classification_method()

  if 'feature_ranking_method' in kwargs and 'estimator_for_rfe_ranking' not in kwargs:
    if kwargs['feature_ranking_method'].__name__ == 'perform_RFE_ranking':
      kwargs['estimator_for_rfe_ranking'] = sklearn.base.clone(classification_model)

  if 'classifier_for_feature_selection' not in kwargs:
    kwargs['classifier_for_feature_selection'] = sklearn.base.clone(classification_model)

  if plot_pca:
    print(f'Performing pca for {experiment_name} with {utils.map_items_to_names(pre_processing_methods)}')
    perform_pca(df, sample_label_dict = sample_label_dict, pre_processing_methods = pre_processing_methods, label_dict_for_plotting = None, samples_to_use = None,
                                 title = f'PCA for {experiment_name} with {utils.map_items_to_names(pre_processing_methods)}', **kwargs)

  start_time = time.time()
  loocv_results = run_pipeline_on_loocv_folds(df, sample_label_dict=sample_label_dict, train_folds_df=train_folds_df, test_folds_df=test_folds_df,
                                         info_df= info_df, loocv_fold_indices=loocv_fold_indices, pre_processing_methods = pre_processing_methods,
                                         feature_selection_method = feature_selection_method, classification_model = classification_model, **kwargs)
  end_time = time.time()
  print(f'LOOCV pipeline finished in {end_time - start_time} seconds')
  kwargs['time_taken'] = end_time - start_time
  
  methods_used = {**pre_processing_methods, 'feature_selection_method' : feature_selection_method, 'classification_method' : classification_method, **kwargs}
  method_names = utils.map_items_to_names(methods_used)
  loocv_metrics_row = calculate_loocv_metrics(loocv_results, experiment_name = experiment_name, **method_names)



  return loocv_metrics_row, loocv_results