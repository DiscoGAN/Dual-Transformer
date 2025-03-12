import torch
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


# Split dataset into training and validation sets based on given parameters
def cut_dataset(dataset, parameter):

    # Determine the number of training samples
    train_num = int(dataset.shape[0] * parameter.train_pro)
    # Compute the total number of valid data points considering the cycle and interval constraints
    total = dataset.shape[0] - (parameter.cycle_num * parameter.interval + parameter.day_num)
    print(dataset.shape[0])  # Print the total dataset size again

    # Initialize lists to store features and targets
    total_features, targets = [], []
    # Iterate through the dataset to extract features and targets
    for num in range(total):
        features = []
        # Collect historical features for the given time steps
        for i in range(parameter.cycle_num):
            feature = dataset[num + parameter.interval * i, :]
            features.append(feature)
        # Add future features for the given time steps (with modifications for feature 0)
        for j in range(parameter.cycle_add):
            feature = dataset[num + parameter.interval * parameter.cycle_num + j * parameter.interval, :].copy()
            feature[0] = 0  # Modify first feature value
            features.append(feature)
        # Convert features to numpy array and append to list
        features = np.array(features)
        total_features.append(features)
        # Extract corresponding target values
        target = dataset[num + parameter.interval * parameter.cycle_num + parameter.day_num, :]
        targets.append(target)

    # Convert lists to numpy arrays
    total_features = np.array(total_features)
    targets = np.array(targets)
    # Expand dimension of targets to align with features
    targets = np.expand_dims(targets, axis=1)
    # Concatenate features and targets to form the final dataset
    dataset = np.concatenate((total_features, targets), axis=1)

    # Shuffle and convert dataset to Torch tensors based on training/validation mode
    if parameter.is_train:
        np.random.seed(None)  # Reset random seed for randomness
        np.random.shuffle(dataset)  # Shuffle dataset
        dataset = torch.FloatTensor(dataset)
        train_dataset, vali_dataset = dataset[:train_num, :, :], dataset[train_num:, :, :]
    else:
        dataset = torch.FloatTensor(dataset)
        train_dataset = dataset
        vali_dataset = 0  # No separate test dataset in non-training mode

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


# Perform final model testing and evaluation
def final_test(dataset, parameter, model, criterion):

    # Prepare the dataset by splitting it into test data
    fin_dataset, _ = cut_dataset(dataset, parameter)
    # Print dataset size for debugging
    print(fin_dataset.shape[0])
    print(fin_dataset.shape)

    # Calculate number of batches
    num = int(fin_dataset.shape[0] / parameter.batch_size)
    print(num)
    # Adjust dataset size to fit complete batches
    plus = (num + 1) * parameter.batch_size - fin_dataset.shape[0]
    fin_dataset = torch.cat((fin_dataset, fin_dataset[:plus, :, :]), 0)
    # Recalculate number of batches
    num = int(fin_dataset.shape[0] / parameter.batch_size)
    reminder = fin_dataset.shape[0] % parameter.batch_size
    print(reminder)
    print(num)

    # Initialize lists to store losses, predictions, and targets
    fin_losss, results, targets = [], [], []
    # Iterate over each batch
    for c in range(num):    
        # Extract input features and target values for the current batch
        input_x = fin_dataset[c * parameter.batch_size : c * parameter.batch_size + parameter.batch_size, 
                              :parameter.cycle_num + parameter.cycle_add, :]
        target = fin_dataset[c * parameter.batch_size : c * parameter.batch_size + parameter.batch_size, 
                             parameter.cycle_num + parameter.cycle_add, 0]
        # Perform forward pass
        pred = model.forward(input_x)
        pred = pred.squeeze(1)  # Adjust shape to match target

        # Store predictions and targets
        results.extend(pred.detach().numpy().tolist())
        targets.extend(target.tolist())

        # Compute loss
        loss = criterion(pred, target)
        print(loss)
        fin_losss.append(loss.detach().numpy())

    # Remove extra added samples to match the original dataset size
    targets = targets[:fin_dataset.shape[0] - plus]
    results = results[:fin_dataset.shape[0] - plus]

    # Plot the predicted vs. actual results
    plt.plot(targets, label="Actual")
    plt.plot(results, label="Predicted")
    plt.legend()
    plt.show()

    return results, targets


# Generate dataset information including average, minimum, and maximum values
def create_date_dic(dataset, parameter, date, dataset_original):
    
    # Generate a list of years based on the given parameter range
    year_list = [str(i) for i in np.arange(parameter.start_year, parameter.start_year + parameter.total_year)]
    
    # Determine the number of years covered in the dataset
    year_length = int(date[-1][:4]) - int(date[0][:4])
    # Initialize dictionaries to store dataset values
    dataset_dic, dataset_wl = {}, {}
    total_num = len(date)

    # Populate dataset dictionaries with date-based values
    for i in range(total_num):
        dataset_dic[date[i]] = dataset_original[i].tolist()
        dataset_wl[date[i]] = dataset_original[i, 0]    

    # Define month lengths (accounting for leap years)
    month_length = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    init = 0
    month_average = {}
    # Calculate the monthly average water level for each year
    for year_num in range(year_length + 1):
        year = 1981 + year_num
        year_name = str(year)
        # Adjust February's length for leap years
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0 and year % 3200 != 0):
            month_length[1] = 29
        else:
            month_length[1] = 28
        # Compute the average water level for each month
        for month in range(12):
            water_level_set = []
            month_name = str(month + 1).zfill(2)  # Ensure two-digit month format
            for i in range(month_length[month]):
                water_level_set.append(dataset_original[init][0])
                init += 1
            index = year_name + "-" + month_name
            month_average[index] = np.average(np.array(water_level_set))

    # Identify the maximum and minimum monthly water levels across years
    max_dic, min_dic = {}, {}
    month_list = list(month_average.keys())
    for i in range(12):
        values = [month_average[month_list[i + j * 12]] for j in range(year_length + 1)]
        max_average = max(values)
        min_average = min(values)
        mark_max = str(parameter.start_year + values.index(max_average)) + '-' + str(i + 1)
        mark_min = str(parameter.start_year + values.index(min_average)) + '-' + str(i + 1)
        max_dic[mark_max] = max_average
        min_dic[mark_min] = min_average   
    # Compute daily average, maximum, and minimum water levels
    dataset_daily_info, dataset_daily_average, dataset_daily_max, dataset_daily_min = {}, {}, {}, {}

    # Initialize daily records for each unique day of the year (MM-DD format)
    for j in date:
        if j[5:] not in dataset_daily_info:
            dataset_daily_info[j[5:]] = []
    # Sort daily records dictionary
    dataset_daily_info = {key: dataset_daily_info[key] for key in sorted(dataset_daily_info.keys())}
    # Populate daily records with water levels
    for k in dataset_wl.keys():
        dataset_daily_info[k[5:]].append(dataset_wl[k])
    # Compute daily average water level
    for m in dataset_daily_info.keys():
        dataset_daily_average[m] = sum(dataset_daily_info[m]) / len(dataset_daily_info[m])
    # Identify daily maximum and minimum water levels along with corresponding years
    for themax in dataset_daily_info.keys():
        max_val = max(dataset_daily_info[themax])
        dataset_daily_max[themax] = [max_val, year_list[dataset_daily_info[themax].index(max_val)]]
    for themin in dataset_daily_info.keys():
        min_val = min(dataset_daily_info[themin])
        dataset_daily_min[themin] = [min_val, year_list[dataset_daily_info[themin].index(min_val)]]
    # Append daily average water levels to dataset dictionary
    for d in dataset_dic.keys():
        dataset_dic[d].append(dataset_daily_average[d[5:]])

    # Extract moving average values and concatenate with the dataset
    dataset_with_ave = np.array(list(dataset_dic.values()))[parameter.moving_length:, 7]
    dataset_with_ave = dataset_with_ave[:, np.newaxis]
    dataset = np.concatenate((dataset, dataset_with_ave), axis=1)

    return dataset, dataset_daily_max, dataset_daily_min, max_dic, min_dic


# Analyze and plot prediction results on the testing dataset
def test_analysis(date, dataset, dataset_original, max_dic, min_dic, results, targets, parameter, scaler):
    
    # Extract the last portion of results and targets based on test length
    results = np.array(results).reshape(-1, 1)[-parameter.test_length:]
    targets = np.array(targets).reshape(-1, 1)[-parameter.test_length:]
    date = np.array(date)[-parameter.test_length:]
    
    # Extract other dataset features excluding the target variable
    others = dataset[-parameter.test_length:, 1:]

    # Recover the original scale of predictions and targets
    recovery_preds = np.concatenate((results, others), axis=1)
    recovery_targets = np.concatenate((targets, others), axis=1)
    recovery_preds = scaler.inverse_transform(recovery_preds)
    recovery_targets = scaler.inverse_transform(recovery_targets)

    # Extract relevant data for plotting
    plot_preds = recovery_preds[:, 0].tolist()
    plot_targets = recovery_targets[:, 0].tolist()
    plot_original = dataset_original[-parameter.test_length:, 0].tolist()
    plot_average = recovery_preds[:, 7].tolist()

    # Compute Root Mean Square Error (RMSE)
    plot_preds = np.array(plot_preds)
    plot_targets = np.array(plot_targets)
    mse = np.mean(np.square(plot_preds - plot_targets))
    error = np.sqrt(mse)

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(20, 10))
    x_major_locator = MultipleLocator(30)
    ax2 = ax1.twinx()
    ax1.grid()

    # Initialize variables for tracking leap years and date indexing
    years_long, init = [], 0    

    # Process each prediction year
    for i in range(parameter.predict_long):
        year = parameter.predict_year_start + 1
        the_last_day = ['01-31', '02-28', '03-31', '04-30', '05-31', '06-30',
                        '07-31', '08-31', '09-30', '10-31', '11-30', '12-31']

        # Leap year determination (follows standard leap year rules)
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0 and year % 3200 != 0):
            year_long = 366
            the_last_day[1] = '02-29'
        else:
            year_long = 365
        
        # Prepare max/min water level dictionaries for each month
        max_dic_new, min_dic_new = {}, {}
        max_index_list, max_values_list = list(max_dic.keys()), list(max_dic.values())
        min_index_list, min_values_list = list(min_dic.keys()), list(min_dic.values())

        for i in range(12):
            max_dic_new[i] = [max_index_list[i][:4], max_values_list[i]]
            min_dic_new[i] = [min_index_list[i][:4], min_values_list[i]]

        # Plot max/min water levels for each month
        for mon in range(12):
            p_x = date[init:init + int(the_last_day[mon][3:])]
            init += int(the_last_day[mon][3:])
            y1 = [max_dic_new[mon][1] for _ in p_x]
            y2 = [min_dic_new[mon][1] for _ in p_x]
            ax1.plot(p_x, y1, c='r')
            ax2.plot(p_x, [i * 3.281 for i in y1], c='r')
            ax1.plot(p_x, y2, c='g')
            ax2.plot(p_x, [i * 3.281 for i in y2], c='g')
            ax1.text(p_x[2], y1[0] + 0.01, max_dic_new[mon][0], size=10, weight='normal')
            ax1.text(p_x[2], y2[0] - 0.05, min_dic_new[mon][0], size=10, weight='normal')

    # Plot predicted water levels
    ax1.plot(date, plot_preds, label='Predicted Results', linewidth=2)
    ax1.xaxis.set_major_locator(x_major_locator)
    ax2.plot(date, [i * 3.281 for i in plot_preds], linewidth=2)

    # Plot original and moving-averaged water levels
    ax1.plot(date, plot_original, label='Original Data', linewidth=2)
    ax2.plot(date, [j * 3.281 for j in plot_original], linewidth=2)
    ax1.plot(date, plot_targets, label='Moving-Averaged Data', linewidth=2)
    ax2.plot(date, [k * 3.281 for k in plot_targets], linewidth=2)

    # Plot long-term average water levels
    ax1.plot(date, plot_average, '--', label='Long-Term Average', linewidth=2)
    ax2.plot(date, [k * 3.281 for k in plot_average], '--', linewidth=2)

    # Set axis labels and limits
    ax1.set_xlabel('Time', size=15)
    ax1.set_ylabel('Water Level (Meters)', size=15)
    ax2.set_ylabel('Water Level (Feet)', size=15)
    ax1.set_ylim(182.6, 184.2)
    ax2.set_ylim(182.6 * 3.281, 184.2 * 3.281)

    # Add legend and title
    ax1.legend(loc='best', fontsize=12)
    plt.title('Prediction of Water Level on Test Dataset', size=18)

    # Format x-axis labels and display the plot
    plt.gcf().autofmt_xdate()
    plt.show()

    return error, plot_targets, plot_preds