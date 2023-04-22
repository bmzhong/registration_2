import numpy as np
from util.data_util.dataset import PairDataset
from torch.utils.data import DataLoader


def create_train_dataset(dataset_config, config, segmentation_available):
    subdivide_names_dict = {'00': [], '01': [], '10': [], '11': []}
    train_data_pair_names = np.array(dataset_config['train_pair'])
    for name1, name2 in train_data_pair_names:
        if segmentation_available[name1] and segmentation_available[name2]:
            subdivide_names_dict['11'].append([name1, name2])
        elif not segmentation_available[name1] and segmentation_available[name2]:
            subdivide_names_dict['01'].append([name1, name2])
        elif segmentation_available[name1] and not segmentation_available[name2]:
            subdivide_names_dict['10'].append([name1, name2])
        else:
            subdivide_names_dict['00'].append([name1, name2])
    subdivide_dataset_dict = {
        '00': PairDataset(dataset_config, 'train_pair', data_names=subdivide_names_dict['00'],
                          segmentation_available=segmentation_available),
        '01': PairDataset(dataset_config, 'train_pair', data_names=subdivide_names_dict['01'],
                          segmentation_available=segmentation_available),
        '10': PairDataset(dataset_config, 'train_pair', data_names=subdivide_names_dict['10'],
                          segmentation_available=segmentation_available),
        '11': PairDataset(dataset_config, 'train_pair', data_names=subdivide_names_dict['11'],
                          segmentation_available=segmentation_available)
    }
    subdivide_dataloader_dict = {
        '00': DataLoader(subdivide_dataset_dict['00'], batch_size=config['TrainConfig']['batchsize'],
                         shuffle=True if len(subdivide_dataset_dict['00']) > 0 else False),
        '01': DataLoader(subdivide_dataset_dict['01'], batch_size=config['TrainConfig']['batchsize'],
                         shuffle=True if len(subdivide_dataset_dict['01']) > 0 else False),
        '10': DataLoader(subdivide_dataset_dict['10'], batch_size=config['TrainConfig']['batchsize'],
                         shuffle=True if len(subdivide_dataset_dict['10']) > 0 else False),
        '11': DataLoader(subdivide_dataset_dict['11'], batch_size=config['TrainConfig']['batchsize'],
                         shuffle=True if len(subdivide_dataset_dict['11']) > 0 else False)
    }
    reg_batch_generator_train = create_batch_generator(subdivide_dataloader_dict)
    seg_availabilities = ['00', '01', '10', '11']
    seg_train_sampling_weights = [0] + [len(subdivide_dataloader_dict[s]) for s in seg_availabilities[1:]]
    seg_batch_generator_train = create_batch_generator(subdivide_dataloader_dict, seg_train_sampling_weights)
    return reg_batch_generator_train, seg_batch_generator_train


def create_batch_generator(dataloader_subdivided, weights=None):
    """
    Create a batch generator that samples data pairs with various segmentation availabilities.

    Arguments:
        dataloader_subdivided : a mapping from the labels in seg_availabilities to dataloaders
        weights : a list of probabilities, one for each label in seg_availabilities;
                  if not provided then we weight by the number of data items of each type,
                  effectively sampling uniformly over the union of the datasets

    Returns: batch_generator
        A function that accepts a number of batches to sample and that returns a generator.
        The generator will weighted-randomly pick one of the seg_availabilities and
        yield the next batch from the corresponding dataloader.
    """
    seg_availabilities = ['00', '01', '10', '11']
    if weights is None:
        weights = np.array([len(dataloader_subdivided[s]) for s in seg_availabilities])
    weights = np.array(weights)
    weights = weights / weights.sum()
    dataloader_subdivided_as_iterators = {s: iter(d) for s, d in dataloader_subdivided.items()}

    def batch_generator(num_batches_to_sample):
        for _ in range(num_batches_to_sample):
            seg_availability = np.random.choice(seg_availabilities, p=weights)
            try:
                yield next(dataloader_subdivided_as_iterators[seg_availability])
            except StopIteration:  # If dataloader runs out, restart it
                dataloader_subdivided_as_iterators[seg_availability] = \
                    iter(dataloader_subdivided[seg_availability])
                yield next(dataloader_subdivided_as_iterators[seg_availability])

    return batch_generator
