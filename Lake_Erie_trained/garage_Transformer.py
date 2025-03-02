import torch
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from matplotlib.dates import MonthLocator, DateFormatter


# Function to load dataset from a CSV file
def generate_dataset(file_path):

    np.random.seed(0)
    df = pd.read_csv(file_path, header=0, index_col=0)
    date = np.array(df.index).tolist()
    df.index = pd.to_datetime(df.index)
    dataset = df.values

    return date, dataset

# Function to normalize dataset using MinMaxScaler
def reframeDF(dataset, scaler):

    dataset = dataset.astype('float32')
    # normalize features
    scaled = scaler.fit_transform(dataset)
    return scaled

# Function to compute moving average
def cal_moving_ave(dataset, length):
    total_num = dataset.shape[0]
    data_new = []
    for num in range(total_num-length):
        data = np.average(dataset[num:num+length, :], axis=0)
        data_new.append(data)
    
    data_new = np.array(data_new)
    return data_new

# Function to split dataset into training and testing sets
def cut_dataset(dataset, parameter):

    print(dataset.shape[0])
    total_num = dataset.shape[0] - parameter.interval_1*parameter.cycle_num_before - parameter.interval_2*parameter.cycle_num_later
    train_num = int(total_num*parameter.train_pro)
    total_features = []
    for i in range(total_num):
        features = []
        for j in range(parameter.cycle_num_before):
            feature_before = dataset[i + j*parameter.interval_1]
            features.append(feature_before)
        for k in range(parameter.cycle_num_later):
            feature_later = dataset[i + parameter.cycle_num_before*parameter.interval_1 + k*parameter.interval_2]
            features.append(feature_later)
        target = dataset[i + parameter.cycle_num_before*parameter.interval_1 + parameter.cycle_num_later*parameter.interval_2]
        features.append(target)

        total_features.append(features)

    total_features = np.array(total_features)
    print(total_features.shape)

    if parameter.is_train:
        np.random.shuffle(total_features)
        dataset = torch.FloatTensor(total_features)
        train_dataset, test_dataset = dataset[:train_num, :, :], dataset[train_num:, :, :]
    else:
        train_dataset = torch.FloatTensor(total_features)
        test_dataset = 0


    return train_dataset, test_dataset

# Function to prepare prediction data
def generate_prediction_data(date_test, date_original, dataset_test, dataset_original, parameter):
    
    date = np.concatenate((date_original, date_test), axis=0)
    dataset_original_new = np.concatenate((dataset_original, dataset_test), axis=0)
    dataset = np.concatenate((dataset_original, dataset_test), axis=0)
    dataset = cal_moving_ave(dataset, parameter.moving_length)

    return date, dataset_original_new, dataset

# Function to create dataset dictionary with min/max/average values
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
    
    dataset_with_ave = np.array(list(dataset_dic.values()))[parameter.moving_length:, 8]
    dataset_with_ave = dataset_with_ave[:, np.newaxis]
    dataset = np.concatenate((dataset, dataset_with_ave), axis=1)

    return dataset, dataset_daily_max, dataset_daily_min, max_dic, min_dic

# Function to evaluate model predictions
def final_test(dataset, parameter, model_1, model_2, model_3, criterion):

    fin_dataset, _ = cut_dataset(dataset, parameter)
    print(fin_dataset.shape)
    num = int(fin_dataset.shape[0]/parameter.batch_size)
    print(num)
    plus = (num+1)*parameter.batch_size-fin_dataset.shape[0]
    fin_dataset = torch.cat((fin_dataset, fin_dataset[:plus, :, :]), 0)
    num = int(fin_dataset.shape[0]/parameter.batch_size)
    reminder = fin_dataset.shape[0]%parameter.batch_size
    print(reminder)
    print(num)

    fin_losss, results, targets, weights_1, weights_2 = [], [], [], [], []
    
    for c in range(num):    
        input_x_1 = fin_dataset[c*parameter.batch_size:(c+1)*parameter.batch_size, :parameter.cycle_num_before, :parameter.input_size]
        input_x_2 = fin_dataset[c*parameter.batch_size:(c+1)*parameter.batch_size, parameter.cycle_num_before:parameter.cycle_num_before+parameter.cycle_num_later, 1:parameter.input_size] # Excluding Erie lake level
        target = fin_dataset[c*parameter.batch_size:(c+1)*parameter.batch_size, parameter.cycle_num_before+parameter.cycle_num_later, 0]
        print(input_x_1.shape, input_x_2.shape)
        
        weights = model_3.forward(fin_dataset[c*parameter.batch_size:(c+1)*parameter.batch_size, :, :parameter.input_size])
        pred_1, _ = model_1.forward(input_x_1)
        pred_1 = pred_1.squeeze(1)
        pred_2, _ = model_2.forward(input_x_2)
        pred_2 = pred_2.squeeze(1)
        pred = weights[:, 0]*pred_1 + weights[:, 1]*pred_2
        results.extend(pred.detach().numpy().tolist()) 
        targets.extend(target.tolist())
        weights_1.extend(weights[:, 0].detach().numpy().tolist())
        weights_2.extend(weights[:, 1].detach().numpy().tolist())
        # print(pred.shape)
        loss = criterion(pred, target)
        print(loss)
        fin_losss.append(loss.detach().numpy())
    targets = targets[:fin_dataset.shape[0]-plus]
    results = results[:fin_dataset.shape[0]-plus]
    error = np.sqrt(sum(np.square(np.array(targets) - np.array(results)))/len(results))
    print(error)
    weights_1 = weights_1[:fin_dataset.shape[0]-plus]
    weights_2 = weights_2[:fin_dataset.shape[0]-plus]
    plt.plot(targets)
    plt.plot(results)
    plt.show()
    return results, targets, weights_1, weights_2

# Function to analyze test results
def test_analysis(date, dataset, dataset_original, max_dic, min_dic, results, targets, parameter, scaler):
    
    results = np.array(results).reshape(-1, 1)[-parameter.test_length:]
    targets = np.array(targets).reshape(-1, 1)[-parameter.test_length:]
    date = np.array(date)[-parameter.test_length:]
    date = [datetime.strptime(i, '%Y-%m-%d') for i in date]
    
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
    error_pearson = pearsonr(plot_preds, plot_targets)

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    ax1.grid(True, linestyle='--', alpha=0.7)
    years_long, init = [], 0    
    for num in range(parameter.predict_long):
        year = parameter.predict_year_start + 1
        the_last_day = ['01-31', '02-28', '03-31', '04-30', '05-31', '06-30' ,'07-31', '08-31', '09-30', '10-31', '11-30', '12-31']

        if (year % 4 == 0 and year % 100 != 0) or (year % 4 == 0 and year % 400 == 0):
            the_last_day[1] = '02-29'
        
        max_dic_new, min_dic_new = {}, {}
        max_index_list, max_values_list, min_index_list, min_values_list = list(max_dic.keys()), list(max_dic.values()), list(min_dic.keys()), list(min_dic.values())
        
        for i in range(12):
            max_dic_new[i] = [max_index_list[i][:4], max_values_list[i]]
            min_dic_new[i] = [min_index_list[i][:4], min_values_list[i]]
        
        for mon in range(12):
            p_x = date[init:init+int(the_last_day[mon][3:])]
            init += int(the_last_day[mon][3:])
            y1 = [max_dic_new[mon][1] for _ in p_x]
            y2 = [min_dic_new[mon][1] for _ in p_x]
            
            ax1.plot(p_x, y1, c='r')
            ax2.plot(p_x, [i * 3.281 for i in y1], c='r')
            ax1.plot(p_x, y2, c='g')
            ax2.plot(p_x, [i * 3.281 for i in y2], c='g')
            ax1.text(p_x[0], y1[0] + 0.01, max_dic_new[mon][0], size=11, weight='normal')
            ax1.text(p_x[0], y2[0] - 0.06, min_dic_new[mon][0], size=11, weight='normal')
    
    ax1.plot(date, plot_preds, linewidth=3, label='Dual-Transformer results')
    ax2.plot(date, [i*3.281 for i in plot_preds], linewidth=3)
    ax1.plot(date, plot_targets, label='observation') 
    ax2.plot(date, [k*3.281 for k in plot_targets], linewidth=3)
    ax1.plot(date, plot_average, '--', label='long-term average') 
    ax2.plot(date, [k*3.281 for k in plot_average], '--')
    ax1.set_xlabel('Time', size=15, labelpad=10)
    ax1.set_ylabel('Watel level (m)', size=15, labelpad=10)
    ax2.set_ylabel('Water level (feet)', size=15, labelpad=10)
    ax1.set_ylim(173.2, 175.8)
    ax2.set_ylim(173.2*3.281, 175.8*3.281)
    ax1.tick_params(axis='y', labelsize=13)
    ax2.tick_params(axis='y', labelsize=13)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.xaxis.set_major_locator(MonthLocator())
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    start_date = date[0] - timedelta(days=30) 
    end_date = date[-1] + timedelta(days=30)   
    ax1.set_xlim([start_date, end_date])
    legend1 = ax1.legend(loc='upper right', fontsize=15)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(1.0)
    legend1.get_frame().set_linewidth(0)
    plt.title('Prediction of Water Level on Testing Dataset', size=18, pad=10)
    plt.gcf().autofmt_xdate()
    plt.show()

    np.save('./results/plot_preds.npy', plot_preds)
    np.save('./results/plot_targets.npy', plot_targets)

    return error, error_pearson, plot_targets, plot_preds
