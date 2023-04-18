# def write_LPBA40_json():
#     hdf5_path = '../../datasets/LPBA40.h5'
#     h5_file = h5py.File(hdf5_path, 'r')
#     json_data = dict()
#     json_data['dataset_path'] = 'datasets/LPBA40.h5'
#     json_data['dataset_size'] = int(h5_file.attrs['dataset_size'])
#     json_data['image_size'] = h5_file.attrs['image_size'].tolist()
#     json_data['normalize'] = h5_file.attrs['normalize'].tolist()
#     json_data['region_number'] = int(h5_file.attrs['region_number'])
#     if 'label_map' in h5_file.attrs.keys():
#         label_value_map = {str(temp[0]): int(temp[1]) for temp in h5_file.attrs['label_map']}
#         if '0' in label_value_map.keys():
#             label_value_map.pop('0')
#     else:
#         label_value_map = {str(i): i for i in range(1, json_data['region_number'] + 1)}
#     json_data['label_value'] = label_value_map
#     image_names = np.array(list(h5_file.keys()))
#     train, val_test = train_test_split(image_names, train_size=0.7)
#     val, test = train_test_split(val_test, test_size=2 / 3)
#     train = train.tolist()
#     val = val.tolist()
#     test = test.tolist()
#     train_pairs = [[img1, img2] for img1 in train for img2 in train]
#     val_pairs = [[img1, img2] for img1 in val for img2 in val]
#     test_pairs = [[img1, img2] for img1 in test for img2 in test]
#     # json_data['subset'] = dict()
#     json_data['train_size'] = len(train)
#     json_data['val_size'] = len(val)
#     json_data['test_size'] = len(test)
#     json_data['train_pairs_size'] = len(train_pairs)
#     json_data['val_pairs_size'] = len(val_pairs)
#     json_data['test_pairs_size'] = len(test_pairs)
#
#     json_data['train'] = train
#     json_data['val'] = val
#     json_data['test'] = test
#
#     json_data['train_pairs'] = train_pairs
#     json_data['val_pairs'] = val_pairs
#     json_data['test_pairs'] = test_pairs
#
#     with open('../../datasets/json/LPBA40.json', 'w') as f:
#         json.dump(json_data, f, indent=4)