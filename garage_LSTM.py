import torch
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from matplotlib.pyplot import MultipleLocator

# 生成csv数据
def generate_dataset(file_path):

    np.random.seed(0)
    df = pd.read_csv(file_path, header=0, index_col=0)
    df.index = pd.to_datetime(df.index)
    dataset = df.values

    return dataset


def reframeDF(dataset, scaler):

    dataset = dataset.astype('float32')
    # normalize features
    scaled = scaler.fit_transform(dataset)
    return scaled



def cal_moving_ave(dataset, length):
    total_num = dataset.shape[0]
    data_new = []
    for num in range(total_num-length):
        data = np.average(dataset[num:num+length, :], axis=0)
        data_new.append(data)
    
    data_new = np.array(data_new)
    return data_new


def cut_dataset_cycle(dataset, cycle_num, day_num, pro, is_train, intervel):

    train_num = int(dataset.shape[0]*pro)
    data_length = dataset.shape[0]
    total_features = []
    for num in range(data_length - cycle_num * day_num - intervel):
        features = []
        for i in range(cycle_num):
            features.append(dataset[num+i*day_num, :])
        features.append(dataset[num+cycle_num*day_num + intervel, :])

        total_features.append(features)

    total_features = np.array(total_features)
    if is_train:
        np.random.shuffle(total_features)
        dataset = torch.FloatTensor(total_features)
        train_dataset, test_dataset = dataset[:train_num, :, :], dataset[train_num:, :, :]
        print(train_dataset.shape, test_dataset.shape)
    else:
        train_dataset = torch.FloatTensor(total_features)
        test_dataset = 0

    return train_dataset, test_dataset


# 进行训练集和测试集的划分

def generate_loader(data, batch_size):
    
    load = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    dataiter = iter(load)

    return dataiter


class LSTModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, seq_length, num_layers, dropout_prob=0.5, bias=True):
        super(LSTModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_direction = 1
        self.num_layers = num_layers
        
        # LSTM layer with Dropout
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0, bias=bias)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):

        h_0 = torch.zeros(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(x.device)

        output, _ = self.lstm(x, (h_0, c_0))
        # Apply dropout after LSTM
        output = self.dropout(output)
        # Decode the hidden state of the last time step
        output = output[:, -1, :]
        pred = self.fc(output)

        return pred



def R2(target, pred):

    pred = np.array(pred)
    target = np.array(target)
    target_average = np.average(target)
    up = np.sum(np.square(target - pred))
    down = np.sum(np.square(pred - target_average))
    r2 = 1 - (up/down)

    return r2


def generate_batch(data, batch_size):
    batch_num = int(data.shape[0]//batch_size)
    batch_set = []
    for i in range(batch_num):
        batch_set.append(data[i*batch_size:(i+1)*batch_size])
    return batch_num, batch_set


def trian(dataset, parameter, model, criterion, optimizer):

    train_losss = []
    for epoch in range(parameter.num_epoch):
        train_dataset, test_dataset = cut_dataset_cycle(dataset, parameter.cycle_num, parameter.day_num, parameter.train_pro, True, parameter.intervel) 
        print('done')
        batch_num, train_set = generate_batch(train_dataset, parameter.batch_size)
        print('done')
        for i in range(batch_num):
            print('epoch:', epoch, 'iter:', i)
            optimizer.zero_grad()
            train_ele = train_set[i]
            print(train_ele.shape)
            input_x = train_ele[:, :parameter.cycle_num, :]
            print(input_x.shape)
            target = train_ele[:, parameter.cycle_num, 0]
            print(target.shape)
            pred = model.forward(input_x).squeeze(1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            print(loss) 
            value = loss.detach().numpy() 
            train_losss.append(value)
        
    train_losss = np.array(train_losss)
    np.save('./train_losss.npy', train_losss)


def final_test(dataset, parameter, model, criterion):
    
    fin_dataset, _ = cut_dataset_cycle(dataset, parameter.cycle_num, parameter.day_num, parameter.train_pro, False, parameter.intervel)
    print(fin_dataset.shape)
    num = int(fin_dataset.shape[0]/parameter.batch_size)
    print(num)
    plus = (num+1)*parameter.batch_size-fin_dataset.shape[0]
    fin_dataset = torch.cat((fin_dataset, fin_dataset[:plus, :, :]), 0)
    print(fin_dataset.shape)
    num = int(fin_dataset.shape[0]/parameter.batch_size)
    reminder = fin_dataset.shape[0]%parameter.batch_size
    print(reminder)
    print(num)

    fin_losss, results, targets = [], [], []
    
    for c in range(num):
        input_x = fin_dataset[c*parameter.batch_size:c*parameter.batch_size+parameter.batch_size, :parameter.cycle_num, :]
        target = fin_dataset[c*parameter.batch_size:c*parameter.batch_size+parameter.batch_size, parameter.cycle_num, 0]
        pred = model.forward(input_x).squeeze(1)
        results.extend(pred.detach().numpy().tolist())
        targets.extend(target.tolist())
        print(pred.shape)
        loss = criterion(pred, target)
        print(loss)
        fin_losss.append(loss.detach().numpy())

    targets = targets[:fin_dataset.shape[0]-plus]
    results = results[:fin_dataset.shape[0]-plus]
    plt.plot(targets)
    plt.plot(results)
    plt.show()
    return results, targets
    

def continue_predict(dataset, model, parameter, criterion):
    preds, targets = [], []
    fix = init_num = parameter.batch_size+parameter.day_num*(parameter.cycle_num-1)
    expand_data = dataset[:init_num, :]
    total = dataset.shape[0]

    for i in range(total - fix):
        print(i)
        create_batch = []
        for j in range(parameter.batch_size):
            batch = []
            for k in range(parameter.cycle_num):
                day = ((init_num-j)-k*parameter.day_num)-1
#                print(day)
                data = expand_data[day, :]
                batch.append(data)
            create_batch.append(np.array(batch))
        create_batch = np.array(create_batch)
        create_batch = torch.FloatTensor(create_batch)
        pred = model.forward(create_batch).squeeze(1)
        the_append_one = [pred[parameter.batch_size-parameter.day_num].tolist()]
        vector = np.array(the_append_one + dataset[init_num, 1:].tolist()).reshape((1, -1))
        preds.append(the_append_one[0])
        target = [dataset[init_num, 0]]
        targets.append(target)
        loss = criterion(torch.FloatTensor(target), torch.FloatTensor(the_append_one))
        print(loss)
#        print(vector.shape)
        expand_data = np.concatenate((expand_data, vector), axis=0)
#        print(expand_data.shape)

        init_num = init_num+1
#        print(init_num)
    R2(targets, preds)
    print(len(preds))
    plt.plot(preds)
    plt.plot(targets)
    plt.show()


# 进行训练集和测试集的划分
def create_date_dic(dataset, parameter, date, dataset_original):
    
    year_list = [str(i) for i in np.arange(parameter.start_year, parameter.start_year+parameter.total_year)]
    year_length = int(date[-1][:4]) - int(date[0][:4])
    dataset_dic, dataset_wl = {}, {}
    total_num = len(date)
    for i in range(total_num):
        dataset_dic[date[i]] = dataset_original[i].tolist()
        dataset_wl[date[i]] = dataset_original[i, 0]    

    month_length = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    init = 0
    month_average = {}
    for year_num in range(year_length+1):
        year = 1981 + year_num
        year_name = str(year)
        if (year%4 == 0 and year%100 != 0) or (year%4 == 0 and year%400 == 0) and (year%4 == 0 and year%3200 != 0):
            month_length[1] = 29
        else:
            month_length[1] = 28
        for month in range(12):
            water_level_set = []
            if month >= 9:
                month_name = str(month+1)
            else:
                month_name = '0'+str(month+1)
            for i in range(month_length[month]):
                water_level_set.append(dataset_original[init][0])
                init = init + 1
            index = year_name + "-" + month_name
            month_average[index] = np.average(np.array(water_level_set))
    
    max_dic, min_dic = {}, {}
    month_list = []
    for i in month_average.keys():
        month_list.append(i)
    for i in range(12):
        values = []
        for j in range(year_length+1):
            value = month_average[month_list[i + j * 12]]
            values.append(value)
        
        max_average = max(values)
        min_average = min(values)
        mark_max = str(parameter.start_year+values.index(max(values))) + '-' + str(i+1)
        mark_min = str(parameter.start_year+values.index(min(values))) + '-' + str(i+1)
        max_dic[mark_max] = max_average
        min_dic[mark_min] = min_average
            
    dataset_daily_info, dataset_daily_average, dataset_daily_max, dataset_daily_min = {}, {}, {}, {}
    for j in date:
        if j[5 : ] not in dataset_daily_info:
            dataset_daily_info[j[5: ]] = []
    dataset_daily_info = {key: dataset_daily_info[key] for key in sorted(dataset_daily_info.keys())}
    for k in dataset_wl.keys():
        dataset_daily_info[k[5:]].append(dataset_wl[k])
    for m in dataset_daily_info.keys():
        dataset_daily_average[m] = sum(dataset_daily_info[m])/len(dataset_daily_info[m])
    for themax in dataset_daily_info.keys():
        dataset_daily_max[themax] = [max(dataset_daily_info[themax]), year_list[dataset_daily_info[themax].index(max(dataset_daily_info[themax]))]]
    for themin in dataset_daily_info.keys():
        dataset_daily_min[themin] = [min(dataset_daily_info[themin]), year_list[dataset_daily_info[themin].index(min(dataset_daily_info[themin]))]]

    for d in dataset_dic.keys():
        dataset_dic[d].append(dataset_daily_average[d[5:]])
    
    dataset_with_ave = np.array(list(dataset_dic.values()))[parameter.moving_length:, 7]
    dataset_with_ave = dataset_with_ave[:, np.newaxis]
    dataset = np.concatenate((dataset, dataset_with_ave), axis=1)

    return dataset, dataset_daily_max, dataset_daily_min, max_dic, min_dic


def test_analysis(date, dataset, dataset_original, max_dic, min_dic, results, targets, parameter, scaler):
    
    results = np.array(results).reshape(-1, 1)[-parameter.test_length:]
    targets = np.array(targets).reshape(-1, 1)[-parameter.test_length:]
    date = np.array(date)[-parameter.test_length:]
    
    others = dataset[-parameter.test_length:, 1:]

    recovery_preds = np.concatenate((results, others), axis=1)
    recovery_targets = np.concatenate((targets, others), axis=1)
    recovery_preds = scaler.inverse_transform(recovery_preds)
    recovery_targets = scaler.inverse_transform(recovery_targets)
    plot_preds = recovery_preds[:, 0].tolist()
    plot_targets = recovery_targets[:, 0].tolist()
    plot_original = dataset_original[-parameter.test_length:, 0].tolist()
    plot_average = recovery_preds[:, 7].tolist()
    error = np.array(plot_preds) - np.array(plot_targets)
    print(error.shape)
    error = np.sqrt(sum(np.square(error))/parameter.test_length)

    fig, ax1 = plt.subplots(figsize=(20, 10))
    x_major_locator = MultipleLocator(30)
    ax2 = ax1.twinx()
    ax1.grid()
    years_long, init = [], 0    
    for i in range(parameter.predict_long):
        year = parameter.predict_year_start + 1
        the_last_day = ['01-31', '02-28', '03-31', '04-30', '05-31', '06-30' ,'07-31', '08-31', '09-30', '10-31', '11-30', '12-31']
        #  闰年规律：四年一闰，百年不闰，四百年再闰，三千二百年再不闰
        if (year%4 == 0 and year%100 != 0) or (year%4 == 0 and year%400 == 0) and (year%4 == 0 and year%3200 != 0):
            year_long = 366
        else:
            year_long = 365

        if year_long == 365:
            the_last_day = the_last_day
        else:
            the_last_day[1] = '02-29'
        
        max_dic_new, min_dic_new = {}, {}
        max_index_list, max_values_list, min_index_list, min_values_list = list(max_dic.keys()), list(max_dic.values()), list(min_dic.keys()), list(min_dic.values())
        for i in range(12):
            max_dic_new[i] = [max_index_list[i][:4], max_values_list[i]]
            min_dic_new[i] = [min_index_list[i][:4], min_values_list[i]]

        for mon in range(12):
            p_x = date[init:init+int(the_last_day[mon][3:])]
            init = init+int(the_last_day[mon][3:])
            y1 = [max_dic_new[mon][1] for i in p_x]
            y2 = [min_dic_new[mon][1] for i in p_x]
            ax1.plot(p_x, y1, c='r')
            ax2.plot(p_x, [i*3.281 for i in y1], c='r')
            ax1.plot(p_x, y2, c='g')
            ax2.plot(p_x, [i*3.281 for i in y2], c='g')
            ax1.text(p_x[2], y1[0]+0.01, max_dic_new[mon][0], size=10, weight='normal')
            ax1.text(p_x[2], y2[0]-0.05, min_dic_new[mon][0], size=10, weight='normal')
    
    ax1.plot(date, plot_preds, label='predicted results')
    ax1.xaxis.set_major_locator(x_major_locator)
    ax2.plot(date, [i*3.281 for i in plot_preds])
    ax1.plot(date, plot_original, label='original data')
    ax2.plot(date, [j*3.281 for j in plot_original])
    ax1.plot(date, plot_targets, label='moving-averaged data') 
    ax2.plot(date, [k*3.281 for k in plot_targets])
    ax1.plot(date, plot_average, '--', label='the long-term average') 
    ax2.plot(date, [k*3.281 for k in plot_average], '--')
    ax1.set_xlabel('Time', size=15)
    ax1.set_ylabel('Meters', size=15)
    ax2.set_ylabel('Feet', size=15)
    ax1.set_ylim(182.6, 184.2)
    ax2.set_ylim(182.6*3.281, 184.2*3.281)
    ax1.legend(loc='best')
    plt.title('Prediction of Water Level on Test Dataset', size=18)
    plt.gcf().autofmt_xdate()
    plt.show()

    return error, plot_targets, plot_preds


def generate_prediction_data(date_test, date_original, dataset_test, dataset_original, parameter):
    
    date = np.concatenate((date_original, date_test), axis=0)
    dataset_original_new = np.concatenate((dataset_original, dataset_test), axis=0)
    dataset = np.concatenate((dataset_original, dataset_test), axis=0)
    dataset = cal_moving_ave(dataset, parameter.moving_length)
#    dataset = reframeDF(dataset, scaler)

    return date, dataset_original_new, dataset