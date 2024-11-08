import torch
import math
import numpy as np
import pandas as pd 
from scipy import signal
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates   
from matplotlib.pyplot import MultipleLocator
from model_base import Transformer_half_base
from model_modify import Transformer_half_modify
from model_weighted import auto_weighted
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from matplotlib.dates import MonthLocator, DateFormatter


# 生成csv数据
def generate_dataset(file_path):

    np.random.seed(0)
    df = pd.read_csv(file_path, header=0, index_col=0)
    date = np.array(df.index).tolist()
    df.index = pd.to_datetime(df.index)
    dataset = df.values

    return date, dataset

 
def reframeDF(dataset, scaler):

    dataset = dataset.astype('float32')
    # normalize features
    scaled = scaler.fit_transform(dataset)

    return scaled


# 我们把数据集划分为可循环的样子，第一步为写出单天为周期的7维输入，这里被循环天数为n天
# 第二步写出多天循环，可选择天数，并且可以选择均值循环还是单点数值循环
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


def generate_loader(data, batch_size):
    
    load = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    dataiter = iter(load)

    return dataiter


def R2(target, pred):

    pred = np.array(pred)
    target = np.array(target)
    target_average = np.average(target)
    up = np.sum(np.square(target - pred))
    down = np.sum(np.square(pred - target_average))
    r2 = 1 - (up/down)

    return r2

def cal_moving_ave(dataset, length):
    total_num = dataset.shape[0]
    data_new = []
    for num in range(total_num-length):
        data = np.average(dataset[num:num+length, :], axis=0)
        data_new.append(data)
    
    data_new = np.array(data_new)
    return data_new

def final_test(dataset, parameter, model_1, model_2, model_3, criterion, parameter_list):

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
        input_x_2 = fin_dataset[c*parameter.batch_size:(c+1)*parameter.batch_size, parameter.cycle_num_before:parameter.cycle_num_before+parameter.cycle_num_later, 1:parameter.input_size]
        target = fin_dataset[c*parameter.batch_size:(c+1)*parameter.batch_size, parameter.cycle_num_before+parameter.cycle_num_later, 0]
        print(input_x_1.shape, input_x_2.shape)
        
        weights = model_3.forward(fin_dataset[c*parameter.batch_size:(c+1)*parameter.batch_size, :, :parameter.input_size])
        pred_1, _ = model_1.forward(input_x_1, parameter_list)
        pred_1 = pred_1.squeeze(1)
        pred_2, _ = model_2.forward(input_x_2, parameter_list)
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
    error_cos = consine_relativity(targets, results)
    print(error, error_cos)
    weights_1 = weights_1[:fin_dataset.shape[0]-plus]
    weights_2 = weights_2[:fin_dataset.shape[0]-plus]
    plt.plot(targets)
    plt.plot(results)
    plt.show()
    return results, targets, weights_1, weights_2


def train(dataset, parameter, model_1, model_2, model_3, criterion, optimizer_1, optimizer_2, optimizer_3, parameter_list):

    trian_dataset, test_dataset = cut_dataset(dataset, parameter)  
    
    train_losss = []
    for epoch in range(parameter.num_epoch):
        print(epoch)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        trainiter = generate_loader(trian_dataset, parameter.batch_size)
        
        train = next(trainiter)
#         print(train)
        input_x_1 = train[:, :parameter.cycle_num_before, :]
        input_x_2 = train[:, parameter.cycle_num_before:parameter.cycle_num_before+parameter.cycle_num_later, 1:]
        target = train[:, parameter.cycle_num_before+parameter.cycle_num_later, 0]
        weights = model_3.forward(train)
        print(weights[0, :])
        pred_1, _ = model_1.forward(input_x_1, parameter_list)
        pred_1 = pred_1.squeeze(1)
        pred_2, _ = model_2.forward(input_x_2, parameter_list)
        pred_2 = pred_2.squeeze(1) 
        pred = pred_1*weights[:, 0] + pred_2*weights[:, 1]
        loss = criterion(pred, target)
        loss.backward()
        optimizer_1.step()
        optimizer_2.step()
        optimizer_3.step()
        print(loss) 
        train_losss.append(loss.detach().numpy())

    train_losss = np.array(train_losss)
    np.save('./results/train_losss.npy', train_losss)

    plt.plot(train_losss)
    plt.show()
    plt.cla()


def pay_attention(dataset, parameter, model_1, model_2, model_3, criterion, parameter_list):

    fin_dataset, _ = cut_dataset(dataset, parameter)
    print(fin_dataset.shape[0])
    print(fin_dataset.shape)
    num = int(fin_dataset.shape[0]/parameter.batch_size)
    print(num)
    plus = (num+1)*parameter.batch_size-fin_dataset.shape[0]
    fin_dataset = torch.cat((fin_dataset, fin_dataset[:plus, :, :]), 0)
    num = int(fin_dataset.shape[0]/parameter.batch_size)
    reminder = fin_dataset.shape[0]%parameter.batch_size
    print(reminder)
    print(num)

    fin_losss, results, targets = [], [], []
    attens_base, attens_modify = [], []
    
    for c in range(num):    
        input_x_1 = fin_dataset[c*parameter.batch_size:c*parameter.batch_size+parameter.batch_size, :parameter.cycle_num_before, :parameter.input_size]
        input_x_2 = fin_dataset[c*parameter.batch_size:c*parameter.batch_size+parameter.batch_size, parameter.cycle_num_before:parameter.cycle_num_before+parameter.cycle_num_later , 1:parameter.input_size]
        target = fin_dataset[c*parameter.batch_size:c*parameter.batch_size+parameter.batch_size, parameter.cycle_num_before+parameter.cycle_num_later, 0]
        weights = model_3.forward(fin_dataset[c*parameter.batch_size:(c+1)*parameter.batch_size, :, :parameter.input_size])
        print(weights[0, :])
        pred_1, atten_base = model_1.forward(input_x_1, parameter_list)
        pred_1 = pred_1.squeeze(1)
        pred_2, atten_modify = model_2.forward(input_x_2, parameter_list)
        pred_2 = pred_2.squeeze(1)
        pred = pred_1*weights[:, 0] + pred_2*weights[:, 1]
        attens_base.append(atten_base)
        attens_modify.append(atten_modify)
        results.extend(pred.detach().numpy().tolist())
        targets.extend(target.tolist())
        # print(pred.shape)
        loss = criterion(pred, target)
        print(loss)
        fin_losss.append(loss.detach().numpy())
    targets = targets[:fin_dataset.shape[0]-plus]
    results = results[:fin_dataset.shape[0]-plus]
    cut_num = fin_dataset.shape[0]-plus
    
    return results, targets, attens_base, attens_modify, cut_num


def generate_prediction_data(date_test, date_original, dataset_test, dataset_original, parameter):
    
    date = np.concatenate((date_original, date_test), axis=0)
    dataset_original_new = np.concatenate((dataset_original, dataset_test), axis=0)
    dataset = np.concatenate((dataset_original, dataset_test), axis=0)
    dataset = cal_moving_ave(dataset, parameter.moving_length)
#    dataset = reframeDF(dataset, scaler)

    return date, dataset_original_new, dataset


def analysis(date, dataset, dataset_original, results, targets, parameter, scaler):


    results = np.array(results).reshape(-1, 1)
    targets = np.array(targets).reshape(-1, 1)
#    reminder = (dataset_original.shape[0]-(parameter.moving_length+parameter.cycle_num*parameter.interval+parameter.day_num))%parameter.batch_size
    others = dataset[parameter.cycle_num*parameter.interval+parameter.day_num:, 1:]
    recovery_preds = np.concatenate((results, others), axis=1)
    recovery_preds = np.concatenate((dataset[:parameter.cycle_num*parameter.interval+parameter.day_num, :], recovery_preds), axis=0)
#    recovery_preds = np.concatenate((recovery_preds, dataset[dataset.shape[0]-reminder:, :]), axis=0)
    recovery_targets = np.concatenate((targets, others), axis=1)
    recovery_targets = np.concatenate((dataset[:parameter.cycle_num*parameter.interval+parameter.day_num, :], recovery_targets), axis=0)
#    recovery_targets = np.concatenate((recovery_targets, dataset[dataset.shape[0]-reminder:, :]), axis=0)
    recovery_preds = scaler.inverse_transform(recovery_preds)
    recovery_targets = scaler.inverse_transform(recovery_targets)
    plot_preds = recovery_preds[:, 0].tolist()[parameter.cycle_num*parameter.interval+parameter.day_num:]
    plot_targets = recovery_targets[:, 0].tolist()[parameter.cycle_num*parameter.interval+parameter.day_num:]
    plot_original = dataset_original[parameter.moving_length+parameter.cycle_num*parameter.interval+parameter.day_num:, 0].tolist()
    x = date[parameter.moving_length+parameter.cycle_num*parameter.interval+parameter.day_num:]
    x_label = pd.to_datetime(x).strftime('%Y').tolist()
    num = int(len(x_label) / parameter.space)
    x_subtitute = ['']+['']+[x_label[parameter.space * i] for i in range(num+1)]

    print(x_subtitute)
    fig, ax1 = plt.subplots()
    x_major_locator = MultipleLocator(parameter.space)
    ax2 = ax1.twinx()
    ax1.grid()
    ax1.plot(x, plot_preds, label='predicted results')
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.set_xticklabels(x_subtitute)
    ax2.plot(x, [i*3.281 for i in plot_preds])
    ax1.plot(x, plot_original, label='original data')
    ax2.plot(x, [j*3.281 for j in plot_original])
    ax1.plot(x, plot_targets, label='averaged data') 
    ax2.plot(x, [k*3.281 for k in plot_targets])
    ax1.set_xlabel('Time', size=15)
    ax1.set_ylabel('Meters', size=15)
    
    ax2.set_ylabel('Feet', size=15)
    ax1.legend(loc='best')
    plt.title('Prediction of Water Level', size=18)
    plt.gcf().autofmt_xdate()
    plt.show()
    plt.cla()


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
    np.save('./average_total.npy', np.array(plot_average))
    error = np.array(plot_preds) - np.array(plot_targets)
    print(error.shape)
    error = np.sqrt(sum(np.square(error))/parameter.test_length)
    error_cos = consine_relativity(plot_preds, plot_targets)
    error_pearson = pearsonr(plot_preds, plot_targets)

    fig, ax1 = plt.subplots(figsize=(14, 8))
    # x_major_locator = MultipleLocator(30)
    ax2 = ax1.twinx()
    ax1.grid(True, linestyle='--', alpha=0.7)
    years_long, init = [], 0    
    for num in range(parameter.predict_long):
        year = parameter.predict_year_start + 1
        the_last_day = ['01-31', '02-28', '03-31', '04-30', '05-31', '06-30' ,'07-31', '08-31', '09-30', '10-31', '11-30', '12-31']

        # 闰年判断
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
            ax1.text(p_x[0], y2[0] - 0.05, min_dic_new[mon][0], size=11, weight='normal')
    
    ax1.plot(date, plot_preds, label='Dual-Transformer results')
    # ax1.xaxis.set_major_locator(x_major_locator)
    ax2.plot(date, [i*3.281 for i in plot_preds], linewidth=3)
    # ax1.plot(date, plot_original, label='original data')
    # ax2.plot(date, [j*3.281 for j in plot_original])
    ax1.plot(date, plot_targets, label='observation') 
    ax2.plot(date, [k*3.281 for k in plot_targets], linewidth=3)
    ax1.plot(date, plot_average, '--', label='long-term average') 
    ax2.plot(date, [k*3.281 for k in plot_average], '--')
    ax1.set_xlabel('Time', size=15, labelpad=10)
    ax1.set_ylabel('Watel level (m)', size=15, labelpad=10)
    ax2.set_ylabel('Water level (feet)', size=15, labelpad=10)
    ax1.set_ylim(182.6, 184.4)
    ax2.set_ylim(182.6*3.281, 184.4*3.281)
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

    np.save('./plot_preds.npy', plot_preds)

    return error, error_cos, error_pearson, plot_targets, plot_preds, plot_average, plot_original

def get_results(parameter, parameter_dict, dataset, date_test, date_original, dataset_test, dataset_original, scaler):

    errors = []

    for i in range(parameter.model_num):

        parameter.cycle_num_before = parameter_dict[i][0]
        parameter.cycle_num_later = parameter_dict[i][1]
        parameter.interval_1 = parameter_dict[i][2]
        parameter.interval_2 = parameter_dict[i][3]
        train_dataset, test_dataset = cut_dataset(dataset, parameter)
        print(train_dataset.shape)
        print(parameter.cycle_num_before, parameter.cycle_num_later, parameter.interval_1, parameter.interval_2)
        path_base = parameter.path_base + "%d.pt"%i
        path_modify = parameter.path_modify + "%d.pt"%i
        path_weighted = parameter.path_weighted + "%d.pt"%i
        print(path_base, path_modify)
        transformer_base = Transformer_half_base(parameter_dict[i])
        transformer_modify = Transformer_half_modify(parameter_dict[i])
        aw = auto_weighted(parameter)
        criterion = torch.nn.MSELoss()         
        transformer_base.load_state_dict(torch.load(path_base))
        transformer_base.eval()
        transformer_modify.load_state_dict(torch.load(path_modify))
        transformer_modify.eval()
        aw.load_state_dict(torch.load(path_weighted))
        aw.eval()
        date, dataset_original, dataset_new = generate_prediction_data(date_test, date_original, dataset_test, dataset_original, parameter)
        date_dic = date[parameter.moving_length:]
        #dataset_new, dataset_daily_max, dataset_daily_min, max_dic, min_dic = create_date_dic(dataset_new, parameter, date, dataset_original)
        dataset_new, dataset_daily_average, dataset_daily_max, dataset_daily_min = create_date_dic(dataset_new, parameter, date, dataset_original)
        dataset_new = reframeDF(dataset_new, scaler)
        print(dataset_new.shape)
        results, targets = final_test(dataset_new, parameter, transformer_base, transformer_modify, aw, criterion, parameter_dict[i])
        np.save('./results/results_%d.npy'%i, results)
        np.save('./results/targets_%d.npy'%i, targets)
        error = test_analysis(date, dataset_new, dataset_original, dataset_daily_average, dataset_daily_max, dataset_daily_min, results, targets, parameter, scaler)
        print(error)
        errors.append(error)

    errors = np.array(errors)
    np.save('./results/errors.npy', errors)


def plot(add, date_test, date_original, dataset_test, dataset_original, scaler, parameter, parameter_dict):

    years_long = []
    for year in parameter.plot_year:
        if (year%4 == 0 and year%100 != 0) or (year%4 == 0 and year%400 == 0) and (year%4 == 0 and year%3200 != 0):
            year_long = 366
        else:
            year_long = 365
        years_long.append(year_long)
    sum_day = sum(years_long)
    print(sum_day)

    errors = np.load('./results/errors.npy')
    print(errors)
    date, dataset_original, dataset_new = generate_prediction_data(date_test, date_original, dataset_test, dataset_original, parameter)
    dataset_new, dataset_daily_max, dataset_daily_min, max_dic, min_dic = create_date_dic(dataset_new, parameter, date, dataset_original)
    dataset_new_scaler = reframeDF(dataset_new, scaler)
    date = [datetime.strptime(i, '%Y-%m-%d') for i in date]
    date_dic = date[parameter.moving_length:]
    plot_date = date_dic[-sum_day:]
    plot_date_target = date_dic[-sum_day:-(sum_day-add)]
    plot_target = dataset_new[-sum_day:-(sum_day-add), 0]
    plot_original = dataset_original[-sum_day:-(sum_day-years_long[0]-add), 0]
    plot_ave = dataset_new[-sum_day:, parameter.input_size]


    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = ax1.twinx()
    ax1.grid()
    years_long, init = [], -sum_day

    for num in range(len(parameter.plot_year)):

        year = parameter.plot_start_year+num
        the_last_day = ['01-31', '02-28', '03-31', '04-30', '05-31', '06-30' ,'07-31', '08-31', '09-30', '10-31', '11-30', '12-31']
        if (year%4 == 0 and year%100 != 0) or (year%4 == 0 and year%400 == 0) and (year%4 == 0 and year%3200 != 0):
            year_long = 366
        else:
            year_long = 365

        if year_long == 365:
            the_last_day = the_last_day
        else:
            the_last_day[1] = '02-29'
        years_long.append(year_long)
        
        max_dic_new, min_dic_new = {}, {}
        max_index_list, max_values_list, min_index_list, min_values_list = list(max_dic.keys()), list(max_dic.values()), list(min_dic.keys()), list(min_dic.values())
        for i in range(12):
            max_dic_new[i] = [max_index_list[i][:4], max_values_list[i]]
            min_dic_new[i] = [min_index_list[i][:4], min_values_list[i]]
        if num == 0:
            total = 12
        else:
            total = 4
        for mon in range(total):
            if init+int(the_last_day[mon][3:]) == 0:
                p_x = date[init:]
            else:
                p_x = date[init:init+int(the_last_day[mon][3:])]
            init = init+int(the_last_day[mon][3:])
            y1 = [max_dic_new[mon][1] for i in p_x]
            y2 = [min_dic_new[mon][1] for i in p_x]
            
            ax1.plot(p_x, y1, c='r')
            ax2.plot(p_x, [i*3.281 for i in y1], c='r')
            ax1.plot(p_x, y2, c='g')
            ax2.plot(p_x, [i*3.281 for i in y2], c='g')
            xs = (pd.to_datetime(p_x[2])-pd.Timedelta(days=0))
            print(type(xs))
            ax1.text(xs, y1[0]+0.01, max_dic_new[mon][0], size=13, weight='normal')
            ax1.text(xs, y2[0]-0.06, min_dic_new[mon][0], size=13, weight='normal')
    

    ax1.xaxis.set_major_locator(MonthLocator())
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    print(plot_original[0], plot_target[0], plot_ave[0])
    # ax1.plot(plot_date_target, plot_original, label='original data', c='steelblue')
    ax1.plot(plot_date_target, plot_target, label='observation', c='orange', linewidth=3)
    ax1.plot(plot_date[:485], plot_ave[:485], '--', label='long-term average', c='grey')

    x_fu, y_fu, error_up_y, error_down_y, real = [], [], [], [], []
    for i in range(parameter.input_size):

        error = errors[i]
        parameter_list = parameter_dict[i]
        day = parameter_list[1]*parameter_list[3]+parameter_list[3]
        results = np.load('./results/results_%d.npy'%i)
        targets = np.load('./results/targets_%d.npy'%i)
        print(results-targets)
        results = np.array(results).reshape(-1, 1)[-sum_day:]
        targets = np.array(targets).reshape(-1, 1)[-sum_day:]
        others = dataset_new_scaler[-sum_day:, 1:]
        recovery_preds = np.concatenate((results, others), axis=1)
        recovery_targets = np.concatenate((targets, others), axis=1)
        recovery_preds = scaler.inverse_transform(recovery_preds)
        recovery_targets = scaler.inverse_transform(recovery_targets)
        results = recovery_preds[:, 0].tolist()
        targets = recovery_targets[:, 0].tolist()
        # print(len(plot_preds), len(plot_targets))
        x_fu.append(date[-sum_day+day+add])
        # ax1.scatter(date[-sum_day+years_long[0]+day+add], results[-sum_day+years_long[0]+day+add], c='green', s=30)
        y_fu.append(results[-sum_day+day+add])
    #    ax1.scatter(date[-sum_day+years_long[0]+day+add], results[-sum_day+years_long[0]+day+add]-error, c='red', s=20)
        error_down_y.append(results[-sum_day+day+add]-error)
    #    ax1.scatter(date[-sum_day+years_long[0]+day+add], results[-sum_day+years_long[0]+day+add]+error, c='red', s=20)
        error_up_y.append(results[-sum_day+day+add]+error)
        # ax1.scatter(date[-sum_day+years_long[0]+day+add], targets[-sum_day+years_long[0]+day+add], c='orange', s=30)
        real.append(targets[-sum_day+day+add])
        # x_value = date[-sum_day+years_long[0]+day+add]
        # y_start = results[-sum_day+years_long[0]+day+add]-error
        # y_end = results[-sum_day+years_long[0]+day+add]+error

#        ax1.plot([x_value, x_value], [y_start, y_end], '--', c='r')
    ax1.scatter(x_fu, y_fu, label="Dual-Transformer prediction points", c="green", s=40)
    ax1.plot(x_fu, y_fu, '--', c="green", linewidth=3)  
    ax2.plot(x_fu, [i*3.281 for i in y_fu], '--', c='green')
    #ax1.plot(x_fu, error_up_y, '--', c='red', label='limitation')  
    #ax1.plot(x_fu, error_down_y, '--', c='red')  
    ax1.scatter(x_fu, real, c='orange', label='observation points', s=40)
    ax1.plot(x_fu, real, '--', c='orange', linewidth=3)
    ax1.set_xlabel('Time', size=15, labelpad=10)
    ax1.set_ylabel('Watel level (m)', size=15, labelpad=10)
    ax2.set_ylabel('Water level (feet)', size=15, labelpad=10)
    ax1.set_ylim(182.6, 184.4)
    ax2.set_ylim(182.6*3.281, 184.4*3.281)
    ax1.tick_params(axis='y', labelsize=13)
    ax1.tick_params(axis='y', labelsize=13)
    ax2.tick_params(axis='y', labelsize=13)
    ax1.tick_params(axis='x', labelsize=11)
    legend1 = ax1.legend(loc='upper right', fontsize=14)
    legend1.get_frame().set_facecolor('white')
    plt.title('Prediction Results from Specific Date', size=18, pad=10)
    plt.gcf().autofmt_xdate()
    plt.show()


def get_attention(num, parameter, date_test, date_original, dataset_test, dataset_original, scaler, parameter_dict):

    parameter.cycle_num_before = parameter_dict[num][0]
    parameter.cycle_num_later = parameter_dict[num][1]
    parameter.interval_1 = parameter_dict[num][2]
    parameter.interval_2 = parameter_dict[num][3]
    transformer_base = Transformer_half_base(parameter_dict[num])
    transformer_modify = Transformer_half_modify(parameter_dict[num])
    aw = auto_weighted(parameter)
    criterion = torch.nn.MSELoss()         

    base_path = parameter.path_base + '%d.pt'%num
    modify_path = parameter.path_modify + '%d.pt'%num
    weighted_path = parameter.path_weighted + '%d.pt'%num
    print(base_path, modify_path)
    transformer_base.load_state_dict(torch.load(base_path))
    transformer_base.eval()
    transformer_modify.load_state_dict(torch.load(modify_path))
    transformer_modify.eval()
    aw.load_state_dict(torch.load(weighted_path))
    aw.eval()
    parameter.is_train = False
    date, dataset_original, dataset_new = generate_prediction_data(date_test, date_original, dataset_test, dataset_original, parameter)
    date_dic = date[parameter.moving_length:]
    dataset_new, dataset_daily_average, dataset_daily_max, dataset_daily_min = create_date_dic(date_dic, dataset_new, parameter)
    dataset_new = reframeDF(dataset_new, scaler)
    print(dataset_new.shape)
    results, targets, attens_base, attens_modify, cut_num = pay_attention(dataset_new, parameter, transformer_base, transformer_modify, aw, criterion, parameter_dict[num])
    print(len(results), len(targets), len(attens_base), len(attens_modify))
    
    attens_base_numpy, attens_modify_numpy = [], []
    for i in range(len(attens_base)):
        atten_base_numpy, atten_modify_numpy = [], []
        for j in range(len(attens_base[i])):
            atten_base_numpy.append(attens_base[i][j].detach().numpy())
            atten_modify_numpy.append(attens_modify[i][j].detach().numpy())
        attens_base_numpy.append(atten_base_numpy)
        attens_modify_numpy.append(atten_modify_numpy)
        
    attens_base = np.array(attens_base_numpy)
    attens_modify = np.array(attens_modify_numpy)
    cut_num = np.array(cut_num)
    
    
    return attens_base, attens_modify, cut_num


def analysis_attention(attens_base, attens_modify, cut_num, attention_year):
    
    base_init, modify_init = attens_base[0], attens_modify[0]

    for i in range(attens_base.shape[0]-1):
        base_init = np.concatenate((base_init, attens_base[i+1]), axis=1)
        modify_init = np.concatenate((modify_init, attens_modify[i+1]), axis=1)

    attens_base = base_init[:, :cut_num, :, :, :].transpose(1, 0, 2, 3, 4)
    attens_modify = modify_init[:, :cut_num, :, :, :].transpose(1, 0, 2, 3, 4)
    
    base_init = attens_base[:, :, 0, : ,:]
    modify_init = attens_modify[:, :, 0, :, :]
    for i in range(attens_base.shape[2]-1):
        base_init = np.concatenate((base_init, attens_base[:, :, i+1, :, :]), axis=3)
        modify_init = np.concatenate((modify_init, attens_modify[:, :, i+1, :, :]), axis=3)
    attens_base = base_init
    attens_modify = modify_init

    base_init = attens_base[:, 0, :, :]
    modify_init = attens_modify[:, 0, :, :]
    for i in range(attens_base.shape[1]-1):
        base_init = np.concatenate((base_init, attens_base[:, i+1, :, :]), axis=1)
        modify_init = np.concatenate((modify_init, attens_modify[:, i+1, :, :]), axis=1)
    attens_base = base_init
    attens_modify = modify_init

    print(attens_modify.shape)
    row, colum = 3, 4
    dic_month = {1:"Janurary", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June", 7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}

    month_long = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    row, colum = 12, 3
    for k in range(2):
        if k % 2 == 0:
            add = 12
            save_path_head = './attention_results_base/'
            
        else:
            add = 5
            save_path_head = './attention_results_modify/'
            
        for layer in range(6):
            for head in range(8):
                fig, axes = plt.subplots(nrows = row, ncols = colum, figsize=(5, 20))
                plt.subplots_adjust(wspace=0.2, hspace=0.2)
                for num in range(3):
                    year = attention_year + num
                    daily_base = attens_base[-(365*(num+1)+1):-(365*num+1)]
                    day_num =0
                    for i in range(12):
                        ax1 = axes.flat[i*3+num]
                        day_num = day_num + month_long[i] - 1 
                        # day_num = day_num + 1
                        ax1.imshow(daily_base[day_num, add*layer:add*(layer+1), add*head:add*(head+1)], cmap="Oranges")
                        ax1.set_xticks([])
                        ax1.set_yticks([])
                        month_name = dic_month[i+1]
                        print(month_name)
                        ax1.set_title('%s_%d' %(month_name, year),  fontdict={'size':10})
                if k % 2 == 0:
                    fig.suptitle('Attention Mechanism in Base Transformer', x=0.5, y=0.905, fontdict={'size':20})
                else:
                    fig.suptitle('Attention Mechanism in Modify Transformer', x=0.5, y=0.905, fontdict={'size':20})
                
                plt.savefig(save_path_head+'%d-%d.png'%(layer, head))

    return attens_base, attens_modify


def cal_corr(matrix_1, matrix_2):

    matrix_1_mean = np.sum(matrix_1)/np.size(matrix_1)
    matrix_2_mean = np.sum(matrix_2)/np.size(matrix_2)

    matrix_1 = matrix_1 - matrix_1_mean
    matrix_2 = matrix_2 - matrix_2_mean

    r = (matrix_1*matrix_2).sum()/math.sqrt((matrix_1*matrix_1).sum() * (matrix_2*matrix_2).sum())

    return r


def validate_attention(attens_base, attens_modify, layer, head):

    year_1_base, year_2_base, year_3_base = attens_base[-365:], attens_base[-(365*2):-365], attens_base[-(365*3):-(365*2)]
    year_1_modify, year_2_modify, year_3_modify = attens_modify[-365:], attens_modify[-(365*2):-365], attens_modify[-(365*3):-(365*2)]

    month_long = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    month_dataset_base, month_dataset_modify = [], []
    day_num = 0
    for i in range(12):
        month_data_base, month_data_modify = [], []
        day_num = day_num + month_long[i] - 1
        month_data_base.append(year_1_base[day_num, 12*(layer-1):12*layer, 12*(head-1):12*head])
        month_data_base.append(year_2_base[day_num, 12*(layer-1):12*layer, 12*(head-1):12*head])
        month_data_base.append(year_3_base[day_num, 12*(layer-1):12*layer, 12*(head-1):12*head])
        month_data_modify.append(year_1_modify[day_num, 5*(layer-1):5*layer, 5*(head-1):5*head])
        month_data_modify.append(year_2_modify[day_num, 5*(layer-1):5*layer, 5*(head-1):5*head])
        month_data_modify.append(year_3_modify[day_num, 5*(layer-1):5*layer, 5*(head-1):5*head])

        month_dataset_base.append(month_data_base)
        month_dataset_modify.append(month_data_modify)
    
    month_dataset_base = np.array(month_dataset_base)
    month_dataset_modify = np.array(month_dataset_modify)
    print(month_dataset_base.shape, month_dataset_modify.shape)


    month_relativity_base = np.zeros((3, 12, 12))
    for year in range(3):
        for i in range(12):
            for j in range(12):
                month_relativity_base[year, j, i] = cal_corr(month_dataset_base[i, year, :, :], month_dataset_base[j, year, :, :])
    
    month_relativity_modify = np.zeros((3, 12, 12))
    for year in range(3):
        for i in range(12):
            for j in range(12):
                month_relativity_modify[year, j, i] = cal_corr(month_dataset_modify[i, year, :, :], month_dataset_modify[j, year, :, :])  

    year_relativity_base = np.zeros((12, 3))
    for i in range(12):
        for j in range(3):
            year_relativity_base[i, j] = cal_corr(month_dataset_base[i, 0, :, :], month_dataset_base[i, 0+j, :, :])

    year_relativity_modify = np.zeros((12, 3))
    for i in range(12):
        for j in range(3):
            year_relativity_modify[i, j] = cal_corr(month_dataset_modify[i, 0, :, :], month_dataset_modify[i, 0+j, :, :])

    
    return month_relativity_base, month_relativity_modify, year_relativity_base, year_relativity_modify


def model_pre_analysis(dataset, parameter, model_1, model_2, model_3, criterion, parameter_list):

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

    fin_losss, results, targets, results_base, results_modify = [], [], [], [], []
    attens_base, attens_modify = [], []
    
    for c in range(num):    
        input_x_1 = fin_dataset[c*parameter.batch_size:c*parameter.batch_size+parameter.batch_size, :parameter.cycle_num_before, :parameter.input_size]
        input_x_2 = fin_dataset[c*parameter.batch_size:c*parameter.batch_size+parameter.batch_size, parameter.cycle_num_before:parameter.cycle_num_before+parameter.cycle_num_later , 1:parameter.input_size]
        target = fin_dataset[c*parameter.batch_size:c*parameter.batch_size+parameter.batch_size, parameter.cycle_num_before+parameter.cycle_num_later, 0]
        weights = model_3.forward(fin_dataset[c*parameter.batch_size:(c+1)*parameter.batch_size, :, :parameter.input_size])
        print(weights[0, :])
        pred_1, _ = model_1.forward(input_x_1, parameter_list)
        pred_1 = pred_1.squeeze(1)
        pred_2, _ = model_2.forward(input_x_2, parameter_list)
        pred_2 = pred_2.squeeze(1)
        pred = weights[:, 0]*pred_1 + weights[:, 1]*pred_2
        pred_1 = weights[:, 0]*pred_1
        pred_2 = weights[:, 1]*pred_2
        results.extend(pred.detach().numpy().tolist())
        results_base.extend(pred_1.detach().numpy().tolist())
        results_modify.extend(pred_2.detach().numpy().tolist())
        targets.extend(target.tolist())
        # print(pred.shape)
        loss = criterion(pred, target)
        print(loss)
        fin_losss.append(loss.detach().numpy())
    targets = targets[:fin_dataset.shape[0]-plus]
    results = results[:fin_dataset.shape[0]-plus]
    results_base = results_base[:fin_dataset.shape[0]-plus]
    results_modify = results_modify[:fin_dataset.shape[0]-plus]
    plt.plot(targets)
    plt.plot(results)
    plt.show()
    return results, results_base, results_modify, targets


def consine_relativity(list_1, list_2):
    sum_xy, num_x, num_y = 0, 0, 0
    for a, b in zip(list_1, list_2):
        sum_xy += a * b
        num_x += a**2
        num_y += b**2
    if num_x == 0 or num_y == 0:
        return None
    else:
        return sum_xy / (num_y*num_x)**0.5
    

def euclidean_distance(vec1, vec2):
    distance = np.sqrt(np.sum(np.power(vec1 - vec2, 2)))/len(vec1)
    return distance


def results_compare(date, targets, results, results_base, results_modify):

    targets = targets
    results = results
    results_base = results_base
    results_modify = results_modify

    b, a = signal.butter(8, 0.01, 'lowpass')
    results_modify_filted = signal.filtfilt(b, a, results_modify).tolist()
    targets_filted = signal.filtfilt(b, a, targets).tolist()
    valley_base_number, valley_base_index = [], []
    valley_modify_number, valley_modify_index = [], []
    for i in range(len(results_modify_filted)-2):
        if results_modify_filted[i] > results_modify_filted[i+1] and results_modify_filted[i+1] < results_modify_filted[i+2]:
            valley_modify_number.append(results_modify_filted[i+1])
            valley_modify_index.append(date[i+1])
   
    for j in range(len(targets_filted)-10):
        if targets_filted[j] > targets_filted[j+1] and targets_filted[j+1] < targets_filted[j+2]:
            valley_base_number.append(targets_filted[j+1])
            valley_base_index.append(date[j+1])

    min_number_set, min_index_set = [], []
    targets_number, targets_index = [], []

    for i in range(20):
        min_number = min(valley_modify_number)
        min_number_set.append(min_number)
        min_index = valley_modify_index[valley_modify_number.index(min_number)]
        min_index_set.append(min_index)
        valley_modify_number[valley_modify_number.index(min_number)] = 0

        min_number_targets = min(valley_base_number)
        targets_number.append(min_number_targets)
        min_index_targets = valley_base_index[valley_base_number.index(min_number_targets)]
        targets_index.append(min_index_targets)
        valley_base_number[valley_base_number.index(min_number_targets)] = 100
    
#     print(valley_base_number)
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.grid()
    ax1.plot(date, targets, label='observation', linewidth=2)
    ax1.plot(date, results_base, label='Prophet results', linewidth=2)
    ax1.plot(date, results_modify_filted, label='Critic results', linewidth=2)
    
    ax1.scatter(min_index_set, min_number_set, c='red')
    ax1.scatter(targets_index, targets_number, c='red')
    ax1.set_ylabel('Water level (normalized)', fontsize=15, labelpad=10)
    ax1.set_xlabel('Year', fontsize=15, labelpad=10)
    ax1.xaxis.set_major_locator(mdates.YearLocator(1)) 
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  
    start_date = date[0] - timedelta(days=365) 
    end_date = date[-1] + timedelta(days=365)   
    ax1.set_xlim([start_date, end_date])
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', labelsize=13)
    ax1.tick_params(axis='x', labelsize=11)
    fig.tight_layout()
    ax1.legend(loc='upper center', fontsize=15)
    plt.show()