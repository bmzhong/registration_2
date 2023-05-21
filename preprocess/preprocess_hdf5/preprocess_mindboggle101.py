import numpy as np
import pandas as pd
import os
import shutil
path = './result1.csv'

data = pd.read_csv(path, header=0)
moving_list, fixed_list = [], []
dice_dict = dict()
for i in range(data.shape[0] - 2):
    movind_fixed = data.iloc[i, 0]
    moving, fixed = movind_fixed.split("_")
    if dice_dict.get(moving, None) is None:
        dice_dict[moving] = [data.loc[i, 'dice']]
    else:
        dice_dict[moving].append(data.loc[i, 'dice'])
result = []
for key in dice_dict.keys():
    dice_dict[key] = np.mean(dice_dict[key])
    if dice_dict[key] < 0.36:
        result.append(key)
path = r'G:\biomdeical\registration\public_data\MindBoggle101\MindBoggle101_from_official_webset\Mindboggle101_individuals\Mindboggle101_volumes\merge'
for dir_name in os.listdir(path):
    if dir_name in result:
        print(dir_name)
        shutil.rmtree(os.path.join(path,dir_name))
