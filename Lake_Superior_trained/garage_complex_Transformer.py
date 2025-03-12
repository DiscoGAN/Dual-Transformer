import torch
import math
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import MultipleLocator


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
    # normalize features
    scaled = scaler.fit_transform(dataset)

    return scaled

# Split dataset into training and validation sets
# Cut the dataset into training and validation sets based on given parameters
def cut_dataset(dataset, parameter):
    
    # Compute the total number of usable samples after accounting for past and future intervals
    total_num = dataset.shape[0] - parameter.interval_1 * parameter.cycle_num_before - parameter.interval_2 * parameter.cycle_num_later
    # Calculate the number of training samples based on the training proportion
    train_num = int(total_num * parameter.train_pro)
    
    # Initialize a list to store the extracted feature sequences
    total_features = []
    # Iterate over the dataset to extract input features and targets
    for i in range(total_num):
        features = []
        # Extract past observations based on the specified cycle number and interval
        for j in range(parameter.cycle_num_before):
            feature_before = dataset[i + j * parameter.interval_1]
            features.append(feature_before)
        
        # Extract future observations based on the specified cycle number and interval
        for k in range(parameter.cycle_num_later):
            feature_later = dataset[i + parameter.cycle_num_before * parameter.interval_1 + k * parameter.interval_2]
            features.append(feature_later)
        # Extract the target value at the end of the sequence
        target = dataset[i + parameter.cycle_num_before * parameter.interval_1 + parameter.cycle_num_later * parameter.interval_2]
        features.append(target)
        # Append the processed feature sequence to the list
        total_features.append(features)
    # Convert the list of features into a NumPy array
    total_features = np.array(total_features)
    # Print the shape of the processed dataset
    print(total_features.shape)

    # If training mode is enabled, shuffle the dataset and split into training and validation sets
    if parameter.is_train:
        np.random.shuffle(total_features)
        dataset = torch.FloatTensor(total_features)
        train_dataset, vali_dataset = dataset[:train_num, :, :], dataset[train_num:, :, :]
    else:
        # If not in training mode, return the entire dataset as training data
        train_dataset = torch.FloatTensor(total_features)
        vali_dataset = 0

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
def create_date_dic(date, dataset, parameter):
    # Generate a list of years based on the specified start year and total duration
    year_list = [str(i) for i in np.arange(parameter.start_year, parameter.start_year + parameter.total_year)]

    # Initialize dictionaries to store dataset values and water level data
    dataset_dic, dataset_wl = {}, {}
    # Get the total number of data points
    total_num = len(date)
    # Populate the dataset dictionary with date keys and corresponding dataset values
    for i in range(total_num):
        dataset_dic[date[i]] = dataset[i].tolist()
        dataset_wl[date[i]] = dataset[i, 0]
    
    # Initialize dictionaries to store daily statistics
    dataset_daily_info, dataset_daily_average, dataset_daily_max, dataset_daily_min = {}, {}, {}, {}
    # Group water level data by day of the year
    for j in date:
        if j[5:] not in dataset_daily_info:
            dataset_daily_info[j[5:]] = []
    
    # Sort daily data keys to ensure chronological order
    dataset_daily_info = {key: dataset_daily_info[key] for key in sorted(dataset_daily_info.keys())}
    # Populate daily water level data
    for k in dataset_wl.keys():
        dataset_daily_info[k[5:]].append(dataset_wl[k])
    # Compute daily average water levels
    for m in dataset_daily_info.keys():
        dataset_daily_average[m] = sum(dataset_daily_info[m]) / len(dataset_daily_info[m])
    # Compute daily maximum and minimum water levels with corresponding years
    for themax in dataset_daily_info.keys():
        max_value = max(dataset_daily_info[themax])
        max_year = year_list[dataset_daily_info[themax].index(max_value)]
        dataset_daily_max[themax] = [max_value, max_year]
    for themin in dataset_daily_info.keys():
        min_value = min(dataset_daily_info[themin])
        min_year = year_list[dataset_daily_info[themin].index(min_value)]
        dataset_daily_min[themin] = [min_value, min_year]

    # Append daily average water level to the dataset dictionary
    for d in dataset_dic.keys():
        dataset_dic[d].append(dataset_daily_average[d[5:]])
    # Convert the dataset dictionary to a NumPy array
    dataset = np.array(list(dataset_dic.values()))

    return dataset, dataset_daily_average, dataset_daily_max, dataset_daily_min


# Perform final model testing and evaluation
def final_test(dataset, parameter, model_1, model_2, criterion, parameter_list):
    # Split dataset for final testing
    fin_dataset, _ = cut_dataset(dataset, parameter)
    print(fin_dataset.shape)

    # Calculate number of batches
    num = int(fin_dataset.shape[0] / parameter.batch_size)
    print(num)

    # Adjust dataset size to fit batch processing
    plus = (num + 1) * parameter.batch_size - fin_dataset.shape[0]
    fin_dataset = torch.cat((fin_dataset, fin_dataset[:plus, :, :]), 0)

    # Recalculate number of batches
    num = int(fin_dataset.shape[0] / parameter.batch_size)
    reminder = fin_dataset.shape[0] % parameter.batch_size
    print(reminder)
    print(num)

    # Initialize lists to store results, targets, and losses
    fin_losss, results, targets = [], [], []
    attens_base, attens_modify = [], []  # Placeholder for attention maps if needed
    # Iterate over each batch
    for c in range(num):    
        # Extract input sequences for both models
        input_x_1 = fin_dataset[c * parameter.batch_size : c * parameter.batch_size + parameter.batch_size, 
                                :parameter.cycle_num_before, :parameter.input_size]
        input_x_2 = fin_dataset[c * parameter.batch_size : c * parameter.batch_size + parameter.batch_size, 
                                parameter.cycle_num_before : parameter.cycle_num_before + parameter.cycle_num_later, 
                                1:parameter.input_size]
        # Extract target values
        target = fin_dataset[c * parameter.batch_size : c * parameter.batch_size + parameter.batch_size, 
                             parameter.cycle_num_before + parameter.cycle_num_later, 0]
        # Forward pass through the first model
        pred_1, _ = model_1.forward(input_x_1, parameter_list)
        pred_1 = pred_1.squeeze(1)
        # Forward pass through the second model
        pred_2, _ = model_2.forward(input_x_2, parameter_list)
        pred_2 = pred_2.squeeze(1)
        # Compute final prediction as the average of both models
        pred = (pred_1 + pred_2) / 2

        # Store predictions and targets
        results.extend(pred.detach().numpy().tolist())
        targets.extend(target.tolist())

        # Compute loss
        loss = criterion(pred, target)
        print(loss)
        fin_losss.append(loss.detach().numpy())

    # Adjust targets and results to remove extra added samples
    targets = targets[:fin_dataset.shape[0] - plus]
    results = results[:fin_dataset.shape[0] - plus]

    # Plot the predicted vs. actual results
    plt.plot(targets, label="Actual")
    plt.plot(results, label="Predicted")
    plt.legend()
    plt.show()

    return results, targets

# Analyze and plot prediction results on the testing dataset
def test_analysis(date, dataset, dataset_original, dataset_daily_average, dataset_daily_max, dataset_daily_min, results, targets, parameter, scaler):
    
    # Reshape prediction and target results to ensure compatibility
    results = np.array(results).reshape(-1, 1)[-parameter.test_length:]
    targets = np.array(targets).reshape(-1, 1)[-parameter.test_length:]
    date = np.array(date)[-parameter.test_length:]
    
    # Extract additional features from the dataset
    others = dataset[-parameter.test_length:, 1:]
    print(results.shape, targets.shape, date.shape, others.shape)

    # Reverse transformation to obtain actual values
    recovery_preds = np.concatenate((results, others), axis=1)
    recovery_targets = np.concatenate((targets, others), axis=1)
    recovery_preds = scaler.inverse_transform(recovery_preds)
    recovery_targets = scaler.inverse_transform(recovery_targets)

    # Extract relevant values for plotting
    plot_preds = recovery_preds[:, 0].tolist()
    plot_targets = recovery_targets[:, 0].tolist()
    plot_original = dataset_original[-parameter.test_length:, 0].tolist()
    plot_average = recovery_preds[:, 7].tolist()

    # Calculate RMSE error
    error = np.array(plot_preds) - np.array(plot_targets)
    print(error.shape)
    error = np.sqrt(sum(np.square(error)) / parameter.test_length)

    # Create the figure and axis for plotting
    fig, ax1 = plt.subplots(figsize=(20, 10))
    x_major_locator = MultipleLocator(30)  # Set major tick interval for x-axis
    ax2 = ax1.twinx()  # Create secondary y-axis
    ax1.grid()  # Enable grid

    # Initialize variables for handling yearly data
    years_long, init = [], 0    
    for i in range(parameter.predict_long):
        year = parameter.predict_year_start + 1

        # Define the last day of each month
        the_last_day = ['01-31', '02-28', '03-31', '04-30', '05-31', '06-30' ,'07-31', 
                        '08-31', '09-30', '10-31', '11-30', '12-31']
        
        # Adjust for leap years (divisible by 4 but not 100, or divisible by 400, excluding 3200)
        if (year % 4 == 0 and year % 100 != 0) or (year % 4 == 0 and year % 400 == 0) and (year % 4 == 0 and year % 3200 != 0):
            year_long = 366
        else:
            year_long = 365

        # Modify February in leap years
        if year_long == 366:
            the_last_day[1] = '02-29'

        years_long.append(year_long)
        print(len(dataset_daily_average))

        # Store max and min water level data for each month
        max_info, min_info = [], []
        for day in the_last_day:
            max_info.append(dataset_daily_max[day])
            min_info.append(dataset_daily_min[day])

        # Plot max and min water level values for each month
        for mon in range(12):
            p_x = date[init:init + int(the_last_day[mon][3:])]
            init = init + int(the_last_day[mon][3:])
            y1 = [max_info[mon][0] for _ in p_x]
            y2 = [min_info[mon][0] for _ in p_x]
            ax1.plot(p_x, y1, c='r')  # Plot max values in red
            ax2.plot(p_x, [i * 3.281 for i in y1], c='r')  # Convert meters to feet
            ax1.plot(p_x, y2, c='g')  # Plot min values in green
            ax2.plot(p_x, [i * 3.281 for i in y2], c='g')  # Convert meters to feet
            ax1.text(p_x[2], y1[0] + 0.01, max_info[mon][1], size=10, weight='normal')  # Annotate max values
            ax1.text(p_x[2], y2[0] - 0.05, min_info[mon][1], size=10, weight='normal')  # Annotate min values

    # Plot prediction results
    ax1.plot(date, plot_preds, label='predicted results')
    ax1.xaxis.set_major_locator(x_major_locator)  # Set x-axis major locator
    ax2.plot(date, [i * 3.281 for i in plot_preds])  # Convert predictions to feet

    # Plot original data, moving average, and long-term average
    ax1.plot(date, plot_original, label='original data')
    ax2.plot(date, [j * 3.281 for j in plot_original])
    ax1.plot(date, plot_targets, label='moving-averaged data')
    ax2.plot(date, [k * 3.281 for k in plot_targets])
    ax1.plot(date, plot_average, '--', label='the average')
    ax2.plot(date, [k * 3.281 for k in plot_average], '--')

    # Configure axis labels and title
    ax1.set_xlabel('Time', size=15)
    ax1.set_ylabel('Meters', size=15)
    ax2.set_ylabel('Feet', size=15)
    ax1.set_ylim(182.6, 184.2)
    ax2.set_ylim(182.6 * 3.281, 184.2 * 3.281)
    ax1.legend(loc='best')

    # Set plot title and format x-axis labels
    plt.title('Prediction of Water Level on Test Dataset', size=18)
    plt.gcf().autofmt_xdate()
    plt.show()

    return error, plot_targets, plot_preds