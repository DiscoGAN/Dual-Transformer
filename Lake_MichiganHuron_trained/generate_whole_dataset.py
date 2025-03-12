import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 


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
water_level = data_s[:, 0]


water_level = np.concatenate((water_level, prediction_water_level))
print(water_level.shape)
water_level = np.expand_dims(water_level, axis=1)
print(water_level.shape)
dataset = np.load('./results/dataset.npy')
dataset = np.concatenate((water_level, dataset), axis=1)
print(dataset.shape)
dataset[:, [0, 1]] = dataset[:, [1, 0]]

np.save('./results/dataset_whole.npy', dataset)
dataset = np.load('./results/dataset_whole.npy')
print(dataset.shape)
plt.plot(dataset[:, 0])
plt.show()