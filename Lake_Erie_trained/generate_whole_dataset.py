import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 


water_level_mh_pre = np.load('./dataset/michiganhuron_water_level_test.npy')

water_level_mh = np.load('./dataset/dataset_mh.npy')[:, 0]
plt.plot(water_level_mh)
water_level_mh[-730:] = water_level_mh_pre
plt.plot(water_level_mh)
water_level_mh = np.expand_dims(water_level_mh, axis=1)
dataset_e = np.load('./results/dataset.npy')
dataset_whole = np.concatenate((water_level_mh, dataset_e), axis=1)
dataset_whole[:, [0, 1]] = dataset_whole[:, [1, 0]]


def generate_dataset(file_path):

    np.random.seed(0)
    df = pd.read_csv(file_path, header=0, index_col=0)
    date = np.array(df.index).tolist()
    df.index = pd.to_datetime(df.index)
    dataset = df.values

    return date, dataset


prediction_water_level = np.load('./dataset/superior_water_level_test.npy')
print(prediction_water_level.shape)

date_s, data_s = generate_dataset('./dataset/finaldata.csv')
water_level_s = data_s[:, 0]
water_level_s = np.concatenate((water_level_s, prediction_water_level))
water_level_s = np.expand_dims(water_level_s, axis=1)
print(water_level_s.shape)
dataset_whole = np.concatenate((water_level_s, dataset_whole), axis=1)
dataset_whole[:, [0, 1]] = dataset_whole[:, [1, 0]]

print(dataset_whole)
np.save('./results/dataset_whole.npy', dataset_whole)
plt.plot(dataset_whole[:,0])
plt.show()