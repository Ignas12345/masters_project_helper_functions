import numpy as np
import pandas as pd

def divide_samples_into_classes(samples, sample_label_dict):
  label_sample_dict = {}
  for label in set(sample_label_dict.values()):
    label_sample_dict[label] = []

  for sample in samples:
    label = sample_label_dict[sample]
    label_sample_dict[label].append(sample)

  return label_sample_dict

def prepare_sample_weights_by_class(sample_label_dict, class_weight_dict: dict = None,  classes_to_use: list = None, fill_unspecified = False):
  '''
  This function takes a data frame and a dictionary of class weights
  and translates that into a dictionary with samples as keys and weights as values.
  If class weight dict is not passed, and only a list of classes to used is passed, then
  the function assigns weights by class sizes.
  If fill unspecified is TRUE, then samples from classes not mentioned in
  class_weight_dict or classes_to_use will be assigned a value of 0.
  '''
  sample_weights = {}
  label_sample_dict = divide_samples_into_classes(sample_label_dict.keys(), sample_label_dict)

  if (classes_to_use is not None) and (class_weight_dict is None):
    class_weight_dict = {}
    for label in classes_to_use:
      class_weight_dict[label] = len(sample_label_dict)/len(label_sample_dict[label])
    min_weight = min(class_weight_dict.values())
    class_weight_dict = {key: value / min_weight for key, value in class_weight_dict.items()}

  for sample, label in sample_label_dict.items():
    if label not in class_weight_dict.keys():
      if fill_unspecified == True:
        sample_weights[sample] = 0
    else:
      sample_weights[sample] = class_weight_dict[label]

  return sample_weights

def make_resample_from_population(samples, desired_fold_size, sample_weights = None, desired_class_proportions_dict = None, sample_label_dict = None, samples_to_avoid = None, with_replacement = False, seed = 42):
  #initialize everything
  rng = np.random.default_rng(seed)
  samples = samples.copy()

  if samples_to_avoid is not None:
    samples = [sample for sample in samples if sample not in samples_to_avoid]
  selected_samples = []

  if desired_class_proportions_dict is not None:
    if sample_label_dict is None:
      raise ValueError('sample_label_dict should be passed if desired_class_proportions_dict is passed')
    labels = desired_class_proportions_dict.keys()
    sum_samps_k = {}
    prob_k = {}

  #now take one sample (possibly together with its copies)
  while len(selected_samples) < desired_fold_size:

    if desired_class_proportions_dict is not None:
    #In this case we first get probabilities of choosing a particular class
      remaining_samples_division_into_classes = divide_samples_into_classes(samples, sample_label_dict)
      selected_samples_division_into_classes = divide_samples_into_classes(selected_samples, sample_label_dict)

      #firstly calculate probability of choosing a given class
      for label in labels:
        sum_samps_k[label] = len(selected_samples_division_into_classes[label])
      sum_upper = sum([sum_samps_k[label] for label in labels])
      sum_lower = sum([desired_class_proportions_dict[label] for label in labels])
      constant = (1 + sum_upper)/sum_lower
      for label in labels:
        prob_k[label] = ((desired_class_proportions_dict[label] * constant) - sum_samps_k[label])

      #secondly sample a class
      for label in prob_k.keys():
        if prob_k[label] < 0:
          prob_k[label] = 0
        elif prob_k[label] > 1:
          prob_k[label] = 1

      chosen_class = rng.choice(list(prob_k.keys()), p = list(prob_k.values()))
      sample_list = remaining_samples_division_into_classes[chosen_class]
    else:
      sample_list = samples

    #now sample from sample list, use weights if they were passed:
    if sample_weights is not None:
      p = [sample_weights[sample] for sample in sample_list]
      p = p/np.sum(p, dtype=float)
      sample_index = rng.choice(len(sample_list), p = p)
    else:
      sample_index = rng.choice(len(sample_list))

    sample = sample_list[sample_index]
    selected_samples += [sample]

    if with_replacement == False:
      samples.remove(sample)

  return selected_samples

def check_for_representitives_in_test_and_train_fold(train_fold, test_fold, smaller_group_samples, larger_group_samples, fold_index, minimum_for_train = 3, minimum_for_test = 1):
  keep_sampling = 1 #if conditions are met, this becomes zero, else remains one and we keep sampling

  if len(set(train_fold).intersection(set(smaller_group_samples))) < minimum_for_train:
      print(f'Not enough samples from smaller group for train fold {fold_index}, rerunning samping')
  elif len(set(test_fold).intersection(set(smaller_group_samples))) < minimum_for_test:
      print(f'Not enough samples in smaller group for test fold {fold_index}, rerunning samping')
  elif len(set(train_fold).intersection(set(larger_group_samples))) < minimum_for_train:
      print(f'Not enough samples from larger group for train fold {fold_index}, rerunning samping')
  elif len(set(test_fold).intersection(set(larger_group_samples))) < minimum_for_test:
      print(f'Not enough samples in larger group for test fold {fold_index}, rerunning samping')
  else:
      keep_sampling = 0

  return keep_sampling

def calculate_fold_proportions(samples, sample_label_dict):
  label_sample_dict = divide_samples_into_classes(samples, sample_label_dict)

  proportion_dict = {}
  for label in label_sample_dict.keys():
    proportion_dict[label] = len(label_sample_dict[label])/len(samples)

  return proportion_dict

def put_folds_into_df(train_folds_df, test_folds_df, index, train_fold, test_fold, classes_to_use, sample_label_dict, comment = None):
  if len(set(train_fold).intersection(set(test_fold))) > 0:
    raise ValueError('Overlapping samples, likely in train and test folds')

  train_folds_df.loc[index, 'samples'] = train_fold
  test_folds_df.loc[index, 'samples'] = test_fold

  train_proportions = calculate_fold_proportions(train_fold, sample_label_dict)
  if len(test_fold) > 0:
    test_proportions = calculate_fold_proportions(test_fold, sample_label_dict)

  train_folds_df.loc[index, f'proportion {classes_to_use[0]}'] = train_proportions[classes_to_use[0]]
  train_folds_df.loc[index, f'proportion {classes_to_use[1]}'] = train_proportions[classes_to_use[1]]
  if len(test_fold) > 0:
    test_folds_df.loc[index, f'proportion {classes_to_use[0]}'] = test_proportions[classes_to_use[0]]
    test_folds_df.loc[index, f'proportion {classes_to_use[1]}'] = test_proportions[classes_to_use[1]]

  train_folds_df.loc[index, 'train_fold_length'] = len(train_fold)
  test_folds_df.loc[index, 'test_fold_length'] = len(test_fold)

  if comment is not None:
    train_folds_df.loc[index, 'comment'] = comment

  return train_folds_df, test_folds_df

def prepare_folds_for_experiment(sample_label_dict, experiment_name):
  print(f'preparing folds for: {experiment_name}')

  group_labels = [label for label in list(set(sample_label_dict.values())) if label != 'unused']
  smaller_group_label = min(group_labels, key=lambda x: len([sample for sample, label in sample_label_dict.items() if label == x]))
  larger_group_label = max(group_labels, key=lambda x: len([sample for sample, label in sample_label_dict.items() if label == x]))
  smaller_group_samples = [sample for sample, label in sample_label_dict.items() if label == smaller_group_label]
  larger_group_samples = [sample for sample, label in sample_label_dict.items() if label == larger_group_label]

  classes_to_use = [smaller_group_label, larger_group_label]

  weights = prepare_sample_weights_by_class(sample_label_dict, classes_to_use = classes_to_use, fill_unspecified=False)

  weight_df = pd.DataFrame.from_dict(weights, orient = 'index', columns=['sample_weight'])
  label_df = pd.DataFrame.from_dict(sample_label_dict, orient = 'index', columns=['label'])
  info_df = pd.concat([weight_df, label_df], axis=1).dropna()

  samples_to_use = info_df.index

  seed = 42
  rng = np.random.default_rng(seed)


  a = len(samples_to_use)
  com_a = 'loocv fold'

  b = 500
  com_b = 'leave-ten-out-cv fold'

  c = 500
  com_c = 'stratified leave-ten-out-cv fold'

  d = 500
  com_d = 'bootstrap fold'
  e = 500
  com_e = 'stratifed bootstrap fold'


  n_folds = 1 + a + b + c + d + e # first fold will be all training samples and no test samples

  train_folds_df = pd.DataFrame(index = [i for i in range(n_folds)], columns = ['samples', f'proportion {classes_to_use[0]}', f'proportion {classes_to_use[1]}', 'train_fold_length', 'comment'])
  test_folds_df = pd.DataFrame(index = [i for i in range(n_folds)], columns = ['samples', f'proportion {classes_to_use[0]}', f'proportion {classes_to_use[1]}', 'test_fold_length'])

  numer_of_folds_discarded = 0

  #first put all training data to folds
  train_fold = samples_to_use
  test_fold = []
  train_folds_df, test_folds_df = put_folds_into_df(train_folds_df, test_folds_df, 0, train_fold, test_fold, classes_to_use, sample_label_dict, 'all training samples')

  #then proceed with loocv folds
  for j in (range(a)):
    i = j + 1
    train_fold = [sample for sample in samples_to_use if sample != samples_to_use[j]]
    test_fold = list(set(samples_to_use) - set(train_fold))
    train_folds_df, test_folds_df = put_folds_into_df(train_folds_df, test_folds_df, i, train_fold, test_fold, classes_to_use, com_a)

  #then with bootstrap an leave-10-out-cv folds
  for j in (range(b)):

    i = j + 1 + a
    keep_sampling = 1
    while keep_sampling:
      samples_to_reserve_for_test_set = []
      samples_to_reserve_for_test_set.append(rng.choice(smaller_group_samples))
      samples_to_reserve_for_test_set.append(rng.choice(larger_group_samples))

      train_fold = make_resample_from_population(samples_to_use, len(samples_to_use) - 10, with_replacement = False, seed = rng.integers(1000000000), samples_to_avoid = samples_to_reserve_for_test_set)
      test_fold = list(set(samples_to_use) - set(train_fold))

      keep_sampling = check_for_representitives_in_test_and_train_fold(train_fold, test_fold, smaller_group_samples, larger_group_samples, i)
      if keep_sampling:
        numer_of_folds_discarded += 1


    train_folds_df, test_folds_df = put_folds_into_df(train_folds_df, test_folds_df, i, train_fold, test_fold, classes_to_use, com_b)
  print(numer_of_folds_discarded)

  #stratified leave_10_out_cv folds
  for j in (range(c)):

    i = j + 1 + a + b
    keep_sampling = 1
    while keep_sampling:
      samples_to_reserve_for_test_set = []
      samples_to_reserve_for_test_set.append(rng.choice(smaller_group_samples))
      samples_to_reserve_for_test_set.append(rng.choice(larger_group_samples))

      train_fold = make_resample_from_population(samples_to_use, len(samples_to_use) - 10, sample_weights=weights, with_replacement = False, seed = rng.integers(1000000000), samples_to_avoid = samples_to_reserve_for_test_set)
      test_fold = list(set(samples_to_use) - set(train_fold))

      keep_sampling = check_for_representitives_in_test_and_train_fold(train_fold, test_fold, smaller_group_samples, larger_group_samples, i)
      if keep_sampling:
        numer_of_folds_discarded += 1

    train_folds_df, test_folds_df = put_folds_into_df(train_folds_df, test_folds_df, i, train_fold, test_fold, classes_to_use, com_c)
  print(numer_of_folds_discarded)

  #bootstrap folds
  for j in (range(d)):

    i = j + 1 + a + b + c
    keep_sampling = 1
    while keep_sampling:
      samples_to_reserve_for_test_set = []
      samples_to_reserve_for_test_set.append(rng.choice(smaller_group_samples))
      samples_to_reserve_for_test_set.append(rng.choice(larger_group_samples))

      train_fold = make_resample_from_population(samples_to_use, len(samples_to_use), with_replacement = True, seed = rng.integers(1000000000), samples_to_avoid = samples_to_reserve_for_test_set)
      test_fold = list(set(samples_to_use) - set(train_fold))

      keep_sampling = check_for_representitives_in_test_and_train_fold(train_fold, test_fold, smaller_group_samples, larger_group_samples, i)
      if keep_sampling:
        numer_of_folds_discarded += 1

    train_folds_df, test_folds_df = put_folds_into_df(train_folds_df, test_folds_df, i, train_fold, test_fold, classes_to_use, com_d)
  print(numer_of_folds_discarded)

  #stratified_bootstrap folds
  for j in (range(e)):

    i = j + 1 + a + b + c + d
    keep_sampling = 1
    while keep_sampling:
      samples_to_reserve_for_test_set = []
      samples_to_reserve_for_test_set.append(rng.choice(smaller_group_samples))
      samples_to_reserve_for_test_set.append(rng.choice(larger_group_samples))

      train_fold = make_resample_from_population(samples_to_use, len(samples_to_use), sample_weights = weights, with_replacement = True, seed = rng.integers(1000000000), samples_to_avoid = samples_to_reserve_for_test_set)
      test_fold = list(set(samples_to_use) - set(train_fold))

      keep_sampling = check_for_representitives_in_test_and_train_fold(train_fold, test_fold, smaller_group_samples, larger_group_samples, i)
      if keep_sampling:
        numer_of_folds_discarded += 1

    train_folds_df, test_folds_df = put_folds_into_df(train_folds_df, test_folds_df, i, train_fold, test_fold, classes_to_use, com_e)
  print(numer_of_folds_discarded)

  return train_folds_df, test_folds_df, info_df