from get_dicts import get_dicts

def return_dicts(tr_type, train, val, test, baseline_directory, num_keypoints=6):
  if tr_type == 'train':
    return get_dicts(baseline_directory, train.index.to_list(), "train", num_keypoints)
  if tr_type == 'val':
    return get_dicts(baseline_directory, val.index.to_list(), "val", num_keypoints)
  if tr_type == 'test':
    return get_dicts(baseline_directory, test.index.to_list(), "test", num_keypoints)