import torch
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from matplotlib.dates import MonthLocator, DateFormatter

# Load dataset from a CSV file
def generate_dataset(file_path):

    np.random.seed(0)
    df = pd.read_csv(file_path, header=0, index_col=0)
    date = np.array(df.index).tolist()
    df.index = pd.to_datetime(df.index)
    dataset = df.values

    return date, dataset


# Compute the moving average of the dataset
def cal_moving_ave(dataset, length):

    total_num = dataset.shape[0]
    data_new = []
    for num in range(total_num-length):
        data = np.average(dataset[num:num+length, :], axis=0)
        data_new.append(data)
    
    data_new = np.array(data_new)

    return data_new


# Normalize dataset using MinMaxScaler 
def reframeDF(dataset, scaler):

    dataset = dataset.astype('float32')
    scaled = scaler.fit_transform(dataset)

    return scaled


# Compute cosine similarity between two lists
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


# Split dataset into training and validation sets
# Cut the dataset into training and validation sets based on given parameters
def cut_dataset(dataset, parameter):

    # Compute the total number of samples after considering intervals and time steps
    total_num = dataset.shape[0] - parameter.interval_1 * parameter.cycle_num_before - parameter.interval_2 * parameter.cycle_num_later
    train_num = int(total_num * parameter.train_pro)  # Determine the number of training samples

    total_features = []  # List to store processed features
    # Construct feature-target pairs for each sample
    for i in range(total_num):
        features = []
        # Extract past features based on interval_1
        for j in range(parameter.cycle_num_before):
            feature_before = dataset[i + j * parameter.interval_1]
            features.append(feature_before)
        # Extract future features based on interval_2
        for k in range(parameter.cycle_num_later):
            feature_later = dataset[i + parameter.cycle_num_before * parameter.interval_1 + k * parameter.interval_2]
            features.append(feature_later)
        # Extract target value
        target = dataset[i + parameter.cycle_num_before * parameter.interval_1 + parameter.cycle_num_later * parameter.interval_2]
        features.append(target)
        total_features.append(features)

    total_features = np.array(total_features)  # Convert list to NumPy array

    # Split dataset into training and validation sets
    if parameter.is_train:
        dataset = total_features
        train_dataset, vali_dataset = dataset[:train_num, :, :], dataset[train_num:, :, :]
        np.random.shuffle(train_dataset)  # Shuffle training data
        train_dataset = torch.FloatTensor(train_dataset)  # Convert to PyTorch tensor
        vali_dataset = torch.FloatTensor(vali_dataset)  # Convert to PyTorch tensor
    else:
        train_dataset = torch.FloatTensor(total_features)  # Convert entire dataset to tensor
        vali_dataset = 0  # No validation dataset if not training

    return train_dataset, vali_dataset


# Generate batches from dataset
def generate_batch(data, batch_size):

    batch_num = int(data.shape[0]//batch_size)
    batch_set = []
    for i in range(batch_num):
        batch_set.append(data[i*batch_size:(i+1)*batch_size])

    return batch_num, batch_set


# Prepare the dataset for the final evaluation in final_test
def generate_prediction_data(date_test, date_original, dataset_test, dataset_original, parameter):
    
    date = np.concatenate((date_original, date_test), axis=0)
    dataset_original_new = np.concatenate((dataset_original, dataset_test), axis=0)
    dataset = np.concatenate((dataset_original, dataset_test), axis=0)
    dataset = cal_moving_ave(dataset, parameter.moving_length)

    return date, dataset_original_new, dataset


# Generate dataset information including average, minimum, and maximum values
def create_date_dic(dataset, parameter, date, dataset_original):

    # Generate a list of years as strings
    year_list = [str(i) for i in np.arange(parameter.start_year, parameter.start_year + parameter.total_year)]
    year_length = int(date[-1][:4]) - int(date[0][:4])  # Calculate the total number of years
    dataset_dic, dataset_wl = {}, {}  # Initialize dictionaries for storing dataset values and water levels
    total_num = len(date)

    # Populate dataset dictionary with original values
    for i in range(total_num):
        dataset_dic[date[i]] = dataset_original[i].tolist()
        dataset_wl[date[i]] = dataset_original[i, 0]  # Store water level data separately

    # Define the number of days in each month
    month_length = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    init = 0
    month_average = {}

    # Compute the monthly average water levels
    for year_num in range(year_length + 1):
        year = 1981 + year_num
        year_name = str(year)
        # Adjust for leap years
        if (year % 4 == 0 and year % 100 != 0) or (year % 4 == 0 and year % 400 == 0) and (year % 4 == 0 and year % 3200 != 0):
            month_length[1] = 29  # February has 29 days in a leap year
        else:
            month_length[1] = 28  # February has 28 days in a non-leap year
        for month in range(12):
            water_level_set = []
            month_name = str(month + 1).zfill(2)  # Format month with leading zero if necessary
            # Collect daily water levels for the given month
            for i in range(month_length[month]):
                water_level_set.append(dataset_original[init][0])
                init += 1
            # Compute monthly average water level
            index = year_name + "-" + month_name
            month_average[index] = np.average(np.array(water_level_set))

    # Compute the maximum and minimum monthly averages across years
    max_dic, min_dic = {}, {}
    month_list = list(month_average.keys())
    for i in range(12):
        values = [month_average[month_list[i + j * 12]] for j in range(year_length + 1)]
        # Get max and min values along with corresponding years
        max_average = max(values)
        min_average = min(values)
        mark_max = str(parameter.start_year + values.index(max_average)) + '-' + str(i + 1)
        mark_min = str(parameter.start_year + values.index(min_average)) + '-' + str(i + 1)
        max_dic[mark_max] = max_average
        min_dic[mark_min] = min_average

    # Compute daily statistics: average, max, and min for each day of the year
    dataset_daily_info, dataset_daily_average, dataset_daily_max, dataset_daily_min = {}, {}, {}, {}
    for j in date:
        if j[5:] not in dataset_daily_info:
            dataset_daily_info[j[5:]] = []
    dataset_daily_info = {key: dataset_daily_info[key] for key in sorted(dataset_daily_info.keys())}
    # Populate daily water level records
    for k in dataset_wl.keys():
        dataset_daily_info[k[5:]].append(dataset_wl[k])
    # Compute daily averages
    for m in dataset_daily_info.keys():
        dataset_daily_average[m] = sum(dataset_daily_info[m]) / len(dataset_daily_info[m])
    # Compute daily max and min values with corresponding years
    for themax in dataset_daily_info.keys():
        dataset_daily_max[themax] = [max(dataset_daily_info[themax]),
                                     year_list[dataset_daily_info[themax].index(max(dataset_daily_info[themax]))]]
    for themin in dataset_daily_info.keys():
        dataset_daily_min[themin] = [min(dataset_daily_info[themin]),
                                     year_list[dataset_daily_info[themin].index(min(dataset_daily_info[themin]))]]
    # Append daily average water levels to the dataset
    for d in dataset_dic.keys():
        dataset_dic[d].append(dataset_daily_average[d[5:]])

    # Extract daily average water levels and concatenate with the dataset
    dataset_with_ave = np.array(list(dataset_dic.values()))[parameter.moving_length:, 7]
    dataset_with_ave = dataset_with_ave[:, np.newaxis]
    dataset = np.concatenate((dataset, dataset_with_ave), axis=1)

    return dataset, dataset_daily_max, dataset_daily_min, max_dic, min_dic


# Perform final model testing and evaluation
def final_test(dataset, parameter, model_1, model_2, model_3, criterion, parameter_list):
    
    # Prepare the dataset for final evaluation
    fin_dataset, _ = cut_dataset(dataset, parameter)
    print(fin_dataset.shape)

    # Compute the number of full batches
    num = int(fin_dataset.shape[0] / parameter.batch_size)
    print(num)

    # Pad dataset to ensure full batch processing
    plus = (num + 1) * parameter.batch_size - fin_dataset.shape[0]
    fin_dataset = torch.cat((fin_dataset, fin_dataset[:plus, :, :]), 0)
    num = int(fin_dataset.shape[0] / parameter.batch_size)
    reminder = fin_dataset.shape[0] % parameter.batch_size
    print(reminder)
    print(num)

    # Initialize lists for storing results
    fin_losss, results, targets = [], [], []
    weights_1, weights_2 = [], []
    results_base, results_modify = [], []
    # Iterate through batches for inference
    for c in range(num):    
        batch_start, batch_end = c * parameter.batch_size, (c + 1) * parameter.batch_size

        # Extract input features and target values
        input_x_1 = fin_dataset[batch_start:batch_end, :parameter.cycle_num_before, :parameter.input_size]
        input_x_2 = fin_dataset[batch_start:batch_end, 
                                parameter.cycle_num_before:parameter.cycle_num_before+parameter.cycle_num_later, 
                                1:parameter.input_size]
        target = fin_dataset[batch_start:batch_end, parameter.cycle_num_before+parameter.cycle_num_later, 0]
        print(input_x_1.shape, input_x_2.shape)
        # Compute weights
        weights = model_3.forward(fin_dataset[batch_start:batch_end, :, :parameter.input_size])

        # Forward pass through the models
        pred_1, _ = model_1.forward(input_x_1, parameter_list)
        pred_1 = pred_1.squeeze(1)
        pred_2, _ = model_2.forward(input_x_2, parameter_list)
        pred_2 = pred_2.squeeze(1)

        # Compute weighted final predictions
        pred = weights[:, 0] * pred_1 + weights[:, 1] * pred_2
        pred_1_weighted = weights[:, 0] * pred_1
        pred_2_weighted = weights[:, 1] * pred_2

        # Store results
        results.extend(pred.detach().numpy().tolist())
        results_base.extend(pred_1_weighted.detach().numpy().tolist())
        results_modify.extend(pred_2_weighted.detach().numpy().tolist())
        targets.extend(target.tolist())
        weights_1.extend(weights[:, 0].detach().numpy().tolist())
        weights_2.extend(weights[:, 1].detach().numpy().tolist())

        # Compute loss
        loss = criterion(pred, target)
        print(loss)
        fin_losss.append(loss.detach().numpy())

    # Remove padded elements to match the original dataset size
    trim_size = fin_dataset.shape[0] - plus
    targets = targets[:trim_size]
    results = results[:trim_size]
    results_base = results_base[:trim_size]
    results_modify = results_modify[:trim_size]
    weights_1 = weights_1[:trim_size]
    weights_2 = weights_2[:trim_size]

    # Compute error metrics
    error = np.sqrt(sum(np.square(np.array(targets) - np.array(results))) / len(results))
    error_cos = consine_relativity(targets, results)
    print(error, error_cos)
    print(len(results_modify), len(targets))

    return results, targets, weights_1, weights_2, results_base, results_modify


# Analyze and plot prediction results on the testing dataset
def test_analysis(date, dataset, dataset_original, max_dic, min_dic, results, targets, parameter, scaler):
    
    # Extract the last test_length data points
    results = np.array(results).reshape(-1, 1)[-parameter.test_length:]
    targets = np.array(targets).reshape(-1, 1)[-parameter.test_length:]
    date = np.array(date)[-parameter.test_length:]
    date = [datetime.strptime(i, '%Y-%m-%d') for i in date]

    # Extract additional features excluding the target variable
    others = dataset[-parameter.test_length:, 1:]

    # Reverse scaling to obtain actual values
    recovery_preds = np.concatenate((results, others), axis=1)
    recovery_targets = np.concatenate((targets, others), axis=1)
    recovery_preds = scaler.inverse_transform(recovery_preds)
    recovery_targets = scaler.inverse_transform(recovery_targets)

    # Extract water level predictions and ground truth
    plot_preds = recovery_preds[:, 0].tolist()
    plot_targets = recovery_targets[:, 0].tolist()
    plot_original = dataset_original[-parameter.test_length:, 0].tolist()
    plot_average = recovery_preds[:, 7].tolist()  # Long-term average water levels
    np.save('./results/results_average.npy', np.array(plot_average))

    # Compute error metrics
    error = np.array(plot_preds) - np.array(plot_targets)
    print(error.shape)
    error = np.sqrt(sum(np.square(error)) / parameter.test_length)
    error_cos = consine_relativity(plot_preds, plot_targets)
    error_pearson = pearsonr(plot_preds, plot_targets)

    # Initialize figure and axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()  # Secondary y-axis for feet conversion
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Define yearly range and initialize variables
    years_long, init = [], 0
    for num in range(parameter.predict_long):
        year = parameter.predict_year_start + 1
        the_last_day = ['01-31', '02-28', '03-31', '04-30', '05-31', '06-30',
                        '07-31', '08-31', '09-30', '10-31', '11-30', '12-31']

        # Adjust for leap years
        if (year % 4 == 0 and year % 100 != 0) or (year % 4 == 0 and year % 400 == 0):
            the_last_day[1] = '02-29'

        # Prepare dictionaries for max/min values per month
        max_dic_new, min_dic_new = {}, {}
        max_index_list, max_values_list = list(max_dic.keys()), list(max_dic.values())
        min_index_list, min_values_list = list(min_dic.keys()), list(min_dic.values())

        for i in range(12):
            max_dic_new[i] = [max_index_list[i][:4], max_values_list[i]]
            min_dic_new[i] = [min_index_list[i][:4], min_values_list[i]]

        # Plot max/min reference lines
        for mon in range(12):
            p_x = date[init:init + int(the_last_day[mon][3:])]
            init += int(the_last_day[mon][3:])
            y1 = [max_dic_new[mon][1] for _ in p_x]
            y2 = [min_dic_new[mon][1] for _ in p_x]

            ax1.plot(p_x, y1, c='r')
            ax2.plot(p_x, [i * 3.281 for i in y1], c='r')
            ax1.plot(p_x, y2, c='g')
            ax2.plot(p_x, [i * 3.281 for i in y2], c='g')

            # Annotate max/min years
            ax1.text(p_x[0], y1[0] + 0.01, max_dic_new[mon][0], size=11, weight='normal')
            ax1.text(p_x[0], y2[0] - 0.05, min_dic_new[mon][0], size=11, weight='normal')

    # Plot predicted and observed water levels
    origin_rmse = np.sqrt(sum(np.square(np.array(plot_preds) - np.array(plot_targets))) / parameter.test_length) * 100
    ax1.plot(date, plot_preds, label='Dual-Transformer results(RMSE=%.1f cm)' % origin_rmse)
    ax2.plot(date, [i * 3.281 for i in plot_preds], linewidth=3)
    ax1.plot(date, plot_targets, label='Observation')
    ax2.plot(date, [k * 3.281 for k in plot_targets], linewidth=3)

    # Plot long-term average water levels
    ax1.plot(date, plot_average, '--', label='Long-term average')
    ax2.plot(date, [k * 3.281 for k in plot_average], '--')

    # Set axis labels
    ax1.set_xlabel('Time', size=15, labelpad=10)
    ax1.set_ylabel('Water level (m)', size=15, labelpad=10)
    ax2.set_ylabel('Water level (feet)', size=15, labelpad=10)

    # Define axis limits
    ax1.set_ylim(182.6, 184.4)
    ax2.set_ylim(182.6 * 3.281, 184.4 * 3.281)

    # Set tick parameters
    ax1.tick_params(axis='y', labelsize=13)
    ax2.tick_params(axis='y', labelsize=13)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.xaxis.set_major_locator(MonthLocator())
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    # Adjust x-axis limits
    start_date = date[0] - timedelta(days=30)
    end_date = date[-1] + timedelta(days=30)
    ax1.set_xlim([start_date, end_date])

    # Configure legend
    legend1 = ax1.legend(loc='upper right', fontsize=15)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(1.0)
    legend1.get_frame().set_linewidth(0)

    # Set plot title
    plt.title('Prediction of Water Level on Testing Dataset', size=18, pad=10)
    plt.gcf().autofmt_xdate()
    plt.show()

    return error, error_cos, error_pearson, plot_targets, plot_preds, plot_average, plot_original