import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def prepare_sample_list(elements, sample_names:list = None, label_sample_dict:dict = None, sample_ordering: np.ndarray = None, sample_label_dict = None):
  '''checks elements against sample_names, a label dictionary and a sample ordering. Returns a list of sample names. At least one of
  sample_names, label_sample_dict or sample_ordering should be provided'''
  samples_to_return = []

  if sample_names is None:
    sample_names = []
  if label_sample_dict is None:
    if sample_label_dict is None:
      label_sample_dict = {}
    else:
      label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')

  for element in elements:
      if element in sample_names:
        samples_to_return.append(element)
      elif element in label_sample_dict:
        samples_to_return += label_sample_dict[element]
      elif type(element) == int:
        if sample_ordering is None:
          raise Exception('sample_ordering not provided, but sample inputted as a numerical index')
        samples_to_return.append(sample_ordering[element])
      else:
        raise Warning('sample ' + element + ' is neither a valid sample, nor a label in the provided arguments - (Check for misspellings or if valid dictionaries, namings are provided)')

  return samples_to_return

def invert_label_dict(dict_to_invert, original_keys: str):
  ''' originl_keys asks if keys of the original dict are 'samples' or 'labels' '''
  inverted_dict = {}

  #go from dict that has samples as keys to a dict that has labels as keys
  if original_keys == 'samples':
    for label in set(dict_to_invert.values()):
      inverted_dict[label] = []
    for sample in dict_to_invert.keys():
      inverted_dict[dict_to_invert[sample]].append(sample)

  #go from dict that has labels as keys to a dict that has values as keys
  if original_keys == 'labels':
    for label in dict_to_invert.keys():
      for sample in dict_to_invert[label]:
        inverted_dict[sample] = label

  return inverted_dict

def create_label_colors(labels:list|dict, color_list = None, default_list = ['blue', 'red', 'orange', 'cyan', 'purple', 'black', 'brown', 'yellow', 'gray']):
  '''labels should be list or sample_label_dict'''
  if(type(labels) == dict):
    label_list = list(set(labels.values()))
  else:
    label_list = labels

  if color_list is None:
    color_list = default_list

  label_color_dict = {}
  for i in range(len(label_list)):
    label_color_dict[label_list[i]] = color_list[i]

  return label_color_dict 

def plot_single_feature(df, feature, samples_to_use:list|None = None, noise_level = 0.05, sample_ordering:list|None = None,
                                  sample_label_dict : dict = None, label_color_dict = None, samples_to_higlight: list|None = None,
                                  xlim = None, ylim = None, show_legend = True, title = None, slice_for_samples:slice = None, use_numbers:bool = None, save_plot = False, random_seed = 42):
  #paima df, kuriu rows yra observazijos ir padaro vieno bruozo plot'a su trupuciu triuksmo.
  #label_dict turėtų turėti sample_ids kaip raktus ir etiketes kaip values
  #slice_for_samples, use_numbers argumentai yra naudojami kitame funckijos variante - ten kur su tekstu plotinna - cia jie nereikalingi

  df = df.copy()
  #add noise column to df
  np.random.seed(random_seed)
  df['noise'] = np.random.normal(loc=0, scale=noise_level, size=len(df))

  if samples_to_use is None:
    samples_to_use = df.index.copy()
    use_all_samples = True
  else:
    use_all_samples = False

  if sample_label_dict is None:
    sample_label_dict = {}
    for sample in samples_to_use:
      sample_label_dict[sample] = 'sample'
  label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')

  if use_all_samples == False:
    samples_to_use = prepare_sample_list(elements = samples_to_use, sample_names = df.index.copy(), label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)

  if label_color_dict is None:
    label_color_dict = create_label_colors(sample_label_dict)
  labels_initial = list(label_color_dict.keys())
  labels = [label for label in labels_initial if label in label_sample_dict.keys()]

  x_data = df[feature]
  y_data = df['noise']
                                    
  if ylim is None:
    y_min = min(x_data)/2
    y_max = max(x_data)/2
    y_diff = y_max - y_min
    y_min = y_min - y_diff/2
    y_max = y_max - y_diff/2
    ylim = [y_min, y_max]

  plt.figure()

  if use_all_samples:
    for label in labels:
      samples_with_label = label_sample_dict[label]
      #plt.scatter(x_data[samples_with_label], y_data[samples_with_label], label = label, c = label_color_dict[label])
      plt.scatter(x_data.loc[samples_with_label], y_data.loc[samples_with_label], label = label, c = label_color_dict[label])
  else:
    for label in labels:
      samples_with_label = list(set(label_sample_dict[label]).intersection(set(samples_to_use)))
      if len(samples_with_label) > 0:
        #plt.scatter(x_data[samples_with_label], y_data[samples_with_label], label = label, c = label_color_dict[label])
        plt.scatter(x_data.loc[samples_with_label], y_data.loc[samples_with_label], label = label, c = label_color_dict[label])

  if samples_to_higlight is not None:
    samples_to_higlight = prepare_sample_list(elements = samples_to_higlight, sample_names = samples_to_use, label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)
    plt.scatter(x_data.loc[samples_to_higlight], y_data.loc[samples_to_higlight], label = 'highlighted', c = 'gold', marker='v')



  if show_legend:
    plt.legend()
  plt.xlabel(feature)
  plt.ylabel('noise of level: ' + str(noise_level))
  if xlim is not None:
    plt.xlim(xlim)
  if ylim is not None:
    plt.ylim(ylim)
  if title is not None:
    plt.title(title)
  if save_plot:
    if title is not None:
      plt.savefig(f'{title}.png')
    else:
      plt.savefig(f'{feature}.png')
  plt.show()

def plot_two_features(df_1, feature_1, feature_2, df_2 = None, samples_to_use:list|None = None, sample_ordering:list|None = None,
                                  sample_label_dict : dict = None, label_color_dict = None, samples_to_higlight: list|None = None,
                                  xlim = None, ylim = None, show_legend = True, title = None, slice_for_samples:slice = None, use_numbers:bool = None, save_plot=False):
  '''
  Paima df, kuriu rows yra observazijos ir padaro dvieju bruozu plot'a.
  label_dict turėtų turėti sample_ids kaip raktus ir etiketes kaip values
  slice_for_samples, use_numbers argumentai yra naudojami kitame funckijos variante - ten kur su tekstu plotinna - cia jie nereikalingi
  Parameters:
  - df_1: DataFrame with features as columns and samples as rows
  - feature_1: name of the first feature to plot
  - feature_2: name of the second feature to plot
  - df_2: DataFrame with features as columns and samples as rows (optional, if not provided, df_1 will be used)
  - samples_to_use: list of samples to use for plotting (optional, if not provided, all samples will be used)
  - sample_ordering: list of samples to use for ordering (optional, if not provided, samples will be used in the order they appear in the DataFrame)
  - sample_label_dict: dictionary with sample ids as keys and labels as values (optional, if not provided, all samples will be labeled as 'sample')
  - label_color_dict: dictionary with labels as keys and colors as values (optional, if not provided, default colors will be used)
  - samples_to_higlight: list of samples to highlight in the plot (optional, if not provided, no samples will be highlighted)
  - xlim: limits for the x-axis (optional)
  - ylim: limits for the y-axis (optional)
  - show_legend: whether to show the legend (default: True)
  - title: title of the plot (optional)
  - slice_for_samples: slice to apply to sample names (optional, not used in this function)
  - use_numbers: whether to use sample numbers instead of names (optional, not used in this function)
  - save_plot: whether to save the plot as a PNG file (default: False)
  '''
  df_1 = df_1.copy()
  if df_2 is None:
    df_2 = df_1.copy()
  else:
    df_2 = df_2.copy()

  if samples_to_use is None:
    samples_to_use = df_1.index.copy()
    use_all_samples = True
  else:
    use_all_samples = False

  if sample_label_dict is None:
    sample_label_dict = {}
    for sample in samples_to_use:
      sample_label_dict[sample] = 'sample'
  label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')

  if use_all_samples == False:
    samples_to_use = prepare_sample_list(elements = samples_to_use, sample_names = df_1.index.copy(), label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)

  if label_color_dict is None:
    label_color_dict = create_label_colors(sample_label_dict)
  labels_initial = list(label_color_dict.keys())
  labels = [label for label in labels_initial if label in label_sample_dict.keys()]

  x_data = df_1[feature_1]
  y_data = df_2[feature_2]

  plt.figure()

  if use_all_samples:
    for label in labels:
      samples_with_label = label_sample_dict[label]
      #plt.scatter(x_data[samples_with_label], y_data[samples_with_label], label = label, c = label_color_dict[label])
      plt.scatter(x_data.loc[samples_with_label], y_data.loc[samples_with_label], label = label, c = label_color_dict[label])
  else:
    for label in labels:
      samples_with_label = list(set(label_sample_dict[label]).intersection(set(samples_to_use)))
      if len(samples_with_label) > 0:
        #plt.scatter(x_data[samples_with_label], y_data[samples_with_label], label = label, c = label_color_dict[label])
        plt.scatter(x_data.loc[samples_with_label], y_data.loc[samples_with_label], label = label, c = label_color_dict[label])

  if samples_to_higlight is not None:
    samples_to_higlight = prepare_sample_list(elements = samples_to_higlight, sample_names = samples_to_use, label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)
    plt.scatter(x_data.loc[samples_to_higlight], y_data.loc[samples_to_higlight], label = 'highlighted', c = 'gold', marker='v')



  if show_legend:
    plt.legend()
  plt.xlabel(feature_1)
  plt.ylabel(feature_2)
  if xlim is not None:
    plt.xlim(xlim)
  if ylim is not None:
    plt.ylim(ylim)
  if title is not None:
    plt.title(title)
  if save_plot:
    if title is not None:
      plt.savefig(f'{title}.png')
    else:
      plt.savefig(f'{feature_1}_{feature_2}.png')
  plt.show()

def plot_single_feature_use_text(df, feature, samples_to_use:list|None = None, noise_level = 0.05, sample_ordering:list|None = None,
                                  sample_label_dict : dict = None, label_color_dict = None, samples_to_higlight: list|None = None,
                                  xlim = None, ylim = None, show_legend = True, title = None, slice_for_samples:slice = None, use_numbers:bool = None, save_plot = False, random_seed=42):
  '''kaip kita plottinim'o funkcija, bet naudoja teksta arba skaicius vietoj tasku.'''
  #paima df, kuriu rows yra observazijos ir padaro dvieju bruozu plot'a.
  #label_dict turėtų turėti sample_ids kaip raktus ir etiketes kaip values

  df = df.copy()
  #add noise column to df
  np.random.seed(random_seed)
  df['noise'] = np.random.normal(loc=0, scale=noise_level, size=len(df))

  if use_numbers and sample_ordering is None:
    raise Exception('sample_ordering should be provided if using numbers to plot')

  if sample_label_dict is None:
    sample_label_dict = {}
    for sample in df.columns:
      sample_label_dict[sample] = 'sample'
  label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')

  if samples_to_use is None:
    samples_to_use = df.index.copy()
  else:
    samples_to_use = prepare_sample_list(elements = samples_to_use, sample_names = df.index.copy(), label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)

  if label_color_dict is None:
    label_color_dict = create_label_colors(sample_label_dict)
  labels_initial = list(label_color_dict.keys())
  labels = [label for label in labels_initial if label in label_sample_dict.keys()]

  x_data = df[feature].copy()
  y_data = df['noise'].copy()
                                    
  if ylim is None:
    y_min = min(y_data)
    y_max = max(y_data)
    buffer = (y_max - y_min)
    ylim = [y_min - buffer, y_max + buffer]

  plt.figure()
  used_labels = []
  patches_for_legend = []

  if xlim is None:
    x_min = min(x_data)
    x_max = max(x_data)
    print('feature "' +feature+ '" ranges from: ' + str(x_min) + ' to ' +str(x_max))
    buffer = (x_max - x_min)/10
    xlim = [x_min - buffer, x_max + buffer]

  for label in labels:
    #used_x = np.array([])
    #used_y = np.array([])
    samples_with_label = list(set(label_sample_dict[label]).intersection(set(samples_to_use)))
    if len(samples_with_label) > 0:
      used_labels += [label]
      x_coords = x_data.loc[samples_with_label].to_numpy()
      y_coords = y_data.loc[samples_with_label].to_numpy()
      #used_x = np.append(used_x, x_coords)
      #used_y = np.append(used_y, y_coords)
      if use_numbers:
        indices = []
        for sample in samples_with_label:
          indices += list(np.where(sample_ordering == sample)[0])
        samples_with_label = indices
      elif slice_for_samples is not None:
        samples_with_label = [sample[slice_for_samples] for sample in samples_with_label]
      for x, y, t in zip(x_coords, y_coords, samples_with_label):
        if x > xlim[0] and x < xlim[1] and y > ylim[0] and y < ylim[1]:
           plt.text(x, y, t, fontsize=10, color= label_color_dict[label])


  for label in used_labels:
    patches_for_legend += [mpatches.Patch(color=label_color_dict[label], label=label)]



  plt.xlim(xlim)
  plt.ylim(ylim)
  if show_legend:
    plt.legend(handles = patches_for_legend)
  plt.xlabel(feature)
  plt.ylabel('noise of level: ' + str(noise_level))
  if title is not None:
    plt.title(title)
  if save_plot:
    if title is not None:
      plt.savefig(f'{title}.png')
    else:
      plt.savefig(f'{feature}.png')
  plt.show()

def plot_two_features_use_text(df_1, feature_1, feature_2, df_2 = None, use_numbers = False, slice_for_samples:slice = None, samples_to_use:list|None = None, sample_ordering:list|None = None,
                                  sample_label_dict : dict = None, label_color_dict = None,
                                  xlim = None, ylim = None, show_legend = True, title = None, save_plot = False):
  '''kaip kita plottinim'o funkcija, bet naudoja teksta arba skaicius vietoj tasku.'''
  #paima df, kuriu rows yra observazijos ir padaro dvieju bruozu plot'a.
  #label_dict turėtų turėti sample_ids kaip raktus ir etiketes kaip values

  '''
  if sample_ordering is not None:
    if df_2 is not None:
      if np.any(df_2.index != sample_ordering):
        print('Warning: df_2 index is not the same as sample_ordering. Better to ensure that the ordering is the same as sample order to avoid incorrect labels or incorrect plotting if using different data frames')

    if np.any(df_1.index != sample_ordering):
      print('Warning: df_1 index is not the same as sample_ordering. Better to ensure that the ordering is the same as sample order to avoid incorrect labels or incorrect plotting if using different data frames')

  #nuo cia assuminam, kad samples abiejose df yra isrikiuoti pagal ta tvarka arba kad meginiu pavadinimai naudojami
  '''
  df_1 = df_1.copy()
  if df_2 is None:
    df_2 = df_1.copy()
  else:
    df_2 = df_2.copy()

  if use_numbers and sample_ordering is None:
    raise Exception('sample_ordering should be provided if using numbers to plot')

  if sample_label_dict is None:
    sample_label_dict = {}
    for sample in df_1.columns:
      sample_label_dict[sample] = 'sample'
  label_sample_dict = invert_label_dict(sample_label_dict, original_keys='samples')

  if samples_to_use is None:
    samples_to_use = df_1.index.copy()
  else:
    samples_to_use = prepare_sample_list(elements = samples_to_use, sample_names = df_1.index.copy(), label_sample_dict = label_sample_dict, sample_ordering = sample_ordering)

  if label_color_dict is None:
    label_color_dict = create_label_colors(sample_label_dict)
  labels_initial = list(label_color_dict.keys())
  labels = [label for label in labels_initial if label in label_sample_dict.keys()]

  x_data = df_1[feature_1].copy()
  y_data = df_2[feature_2].copy()

  plt.figure()
  used_labels = []
  patches_for_legend = []

  if xlim is None:
    x_min = min(x_data)
    x_max = max(x_data)
    print('feature "' +feature_1+ '" ranges from: ' + str(x_min) + ' to ' +str(x_max))
    buffer = (x_max - x_min)/10
    xlim = [x_min - buffer, x_max + buffer]
  if ylim is None:
    y_min = min(y_data)
    y_max = max(y_data)
    print('feature "' +feature_2+ '" ranges from: ' + str(y_min) + ' to ' +str(y_max))
    buffer = (y_max - y_min)/10
    ylim = [y_min - buffer, y_max + buffer]

  for label in labels:
    #used_x = np.array([])
    #used_y = np.array([])
    samples_with_label = list(set(label_sample_dict[label]).intersection(set(samples_to_use)))
    if len(samples_with_label) > 0:
      used_labels += [label]
      x_coords = x_data.loc[samples_with_label].to_numpy()
      y_coords = y_data.loc[samples_with_label].to_numpy()
      #used_x = np.append(used_x, x_coords)
      #used_y = np.append(used_y, y_coords)
      if use_numbers:
        indices = []
        for sample in samples_with_label:
          indices += list(np.where(sample_ordering == sample)[0])
        samples_with_label = indices
      elif slice_for_samples is not None:
        samples_with_label = [sample[slice_for_samples] for sample in samples_with_label]
      for x, y, t in zip(x_coords, y_coords, samples_with_label):
        if x > xlim[0] and x < xlim[1] and y > ylim[0] and y < ylim[1]:
           plt.text(x, y, t, fontsize=10, color= label_color_dict[label])


  for label in used_labels:
    patches_for_legend += [mpatches.Patch(color=label_color_dict[label], label=label)]



  plt.xlim(xlim)
  plt.ylim(ylim)
  if show_legend:
    plt.legend(handles = patches_for_legend)
  plt.xlabel(feature_1)
  plt.ylabel(feature_2)
  if title is not None:
    plt.title(title)
  if save_plot:
    if title is not None:
      plt.savefig(f'{title}.png')
    else:
      plt.savefig(f'{feature_1}_{feature_2}.png')

  plt.show()

  