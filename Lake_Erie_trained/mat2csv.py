import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt 
import pandas as pd 


water_level = loadmat('./dataset/DailyWaterLevel_Erie.mat')['lvl_mat']['water_level'][0][0]
water_level = water_level.reshape((water_level.shape))
print(water_level.shape)

var_list = ['t2_lake', 't2_land', 'wspd_lake', 'wspd_land', 'lst_lake', 'pr_cfsr_basin']
start_date = '19810101'
end_date = '20231231'
date = pd.date_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
date = np.array(date)
np.save('./results/date', date)

final_dataset = []
for i in range(len(var_list)):
    var = var_list[i]
    print(var)
    at_var = loadmat('./dataset/data_gle_1981to2023_cfsr_t2_wspd_precip.mat')['dd'][var][0][0][0]
    year_data_set = []
    for i in range(at_var.shape[0]):
        year_data = at_var[i]
        year_data = list(year_data.reshape((year_data.shape[0])))
        year_data_set.extend(year_data)
    print(len(year_data_set))
    final_dataset.append(year_data_set)

final_dataset = np.array(final_dataset).T
print(final_dataset.shape)
final_dataset = np.concatenate((water_level, final_dataset), axis=1)
np.save('./results/dataset.npy', final_dataset)
print(final_dataset.shape)
print(len(date))

