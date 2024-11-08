import torch
import numpy as np
import matplotlib.pyplot as plt
from garage_LSTM import LSTModel
from garage_LSTM import generate_dataset, reframeDF, cal_moving_ave, R2
from garage_LSTM import trian, final_test, test_analysis, create_date_dic, generate_prediction_data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class config():
    def __init__(self):
        self.cycle_num = 12
        self.day_num = 30
        self.moving_length = 30
        self.is_mean = False
        self.train_pro = 1
        self.input_size = 7
        self.hidden_size = 512
        self.output_size = 1
        self.batch_size = 256
        self.num_layers = 2
        self.learning_rate = 0.0001
        self.num_epoch = 112
        self.path = './lstm_2.pt'
        self.intervel = 150
        self.start_year = 1981
        self.total_year = 100
        self.test_length = 730
        self.predict_long = 2
        self.predict_year_start = 2022


wl_p = config()
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = generate_dataset('./finaldata.csv')
dataset_original = generate_dataset('./finaldata.csv')
dataset_1 = generate_dataset('./test_2022.csv')
dataset_1[:, 1] = dataset_1[:, 1] - 273.15
dataset_1[:, 2] = dataset_1[:, 2] - 273.15
dataset_2 = generate_dataset('./test_2023.csv')
dataset_whole = np.concatenate((dataset, dataset_1))
dataset_whole_original = np.concatenate((dataset_whole, dataset_2))
dataset_whole = np.concatenate((dataset_whole, dataset_2))
dataset_whole = reframeDF(dataset_whole, scaler)
dataset_whole = cal_moving_ave(dataset_whole, wl_p.moving_length)
dataset = reframeDF(dataset, scaler)
dataset = cal_moving_ave(dataset, wl_p.moving_length)
model = LSTModel(wl_p.input_size, wl_p.hidden_size, wl_p.output_size, wl_p.batch_size, wl_p.cycle_num, wl_p.num_layers)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wl_p.learning_rate)



# trian(dataset, wl_p, model, criterion, optimizer)
# torch.save(model.state_dict(), wl_p.path)
model.load_state_dict(torch.load(wl_p.path))
model.eval()
results, targets = final_test(dataset_whole, wl_p, model, criterion)
# np.save('./results.npy', results)
# np.save('./targets.npy', targets)
# results = np.load('./results.npy')
# targets = np.load('./targets.npy')
date = pd.date_range('19810101', '20231231').strftime("%Y-%m-%d").tolist()
date_test, date_original = date[14975:], date[:14975]
dataset_test = dataset_whole_original[14975:]
date, dataset_original, dataset_new = generate_prediction_data(date_test, date_original, dataset_test, dataset_original, wl_p)
dataset_new, dataset_daily_max, dataset_daily_min, max_dic, min_dic = create_date_dic(dataset_new, wl_p, date, dataset_original)
dataset_new = reframeDF(dataset_new, scaler)

error, plot_targets, plot_preds = test_analysis(date, dataset_new, dataset_original, max_dic, min_dic, results, targets, wl_p, scaler)
print(error)
np.save('./lstm_2_results.npy', plot_preds)