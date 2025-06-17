import numpy as np
import matplotlib.pyplot as plt

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
    y_min = y_min - y_diff
    y_max = y_max - y_diff
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