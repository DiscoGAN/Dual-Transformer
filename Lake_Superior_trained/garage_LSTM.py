import torch
import numpy as np
import pandas as pd 
import torch.nn as nn
import matplotlib.pyplot as plt 
from matplotlib.pyplot import MultipleLocator

# Define LSTM model
class LSTModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size, seq_length, num_layers, dropout_prob=0.5, bias=True):
        super(LSTModel, self).__init__()

        # Store model parameters
        self.input_size = input_size  # Number of input features
        self.hidden_size = hidden_size  # Number of hidden units in the LSTM
        self.output_size = output_size  # Number of output features
        self.batch_size = batch_size  # Batch size used for training
        self.seq_length = seq_length  # Sequence length of input data
        self.num_direction = 1  # Unidirectional LSTM (single direction)
        self.num_layers = num_layers  # Number of stacked LSTM layers

        # Define LSTM layer with optional dropout
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0, bias=False)
        
        # Fully connected (linear) layer to generate final output
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state (h_0) and cell state (c_0) with zeros
        h_0 = torch.zeros(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_direction * self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass input through LSTM layer
        output, _ = self.lstm(x, (h_0, c_0))
        # Extract the output from the last time step
        output = output[:, -1, :]
        # Pass LSTM output through the fully connected layer to get predictions
        pred = self.fc(output)
        print('done')  # Debugging statement to indicate completion
        return pred
    
# Load dataset from a CSV file
def generate_dataset(file_path):

    np.random.seed(0)
    df = pd.read_csv(file_path, header=0, index_col=0)
    df.index = pd.to_datetime(df.index)
    dataset = df.values

    return dataset


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
def cut_dataset_cycle(dataset, cycle_num, cycle_add, day_num, pro, is_train, interval):

    # Determine the number of training samples based on the proportion `pro`
    train_num = int(dataset.shape[0] * pro)
    # Calculate total available samples after accounting for cycles and interval
    total = dataset.shape[0] - (cycle_num * day_num + interval)

    # List to store extracted features
    total_features = []
    # Loop through the dataset to extract features
    for num in range(total):
        features = []
        # Extract historical features
        for i in range(cycle_num):
            features.append(dataset[num + i * day_num, :])
        # Extract future features but set the first feature (index 0) to 0
        for j in range(cycle_add):
            feature = dataset[num + day_num * cycle_num + j * day_num, :].copy()
            feature[0] = 0  # Masking the first value to 0 for additional cycles
            features.append(feature)
        # Append the target
        features.append(dataset[num + day_num * cycle_num + interval, :])
        # Store the extracted sequence
        total_features.append(features)
    # Convert extracted features to a NumPy array
    total_features = np.array(total_features)

    if is_train:
        # Shuffle dataset for training
        np.random.shuffle(total_features)
        # Convert dataset to PyTorch FloatTensor
        dataset = torch.FloatTensor(total_features)
        # Split dataset into training and validation sets
        train_dataset, vali_dataset = dataset[:train_num, :, :], dataset[train_num:, :, :]
        print(train_dataset.shape, vali_dataset.shape)  # Debugging output to check dataset shapes
    else:
        # If not training, return entire dataset as `train_dataset`
        train_dataset = torch.FloatTensor(total_features)
        vali_dataset = 0  # No validation set in this case

    return train_dataset, vali_dataset


# Prepare the dataset for the final evaluation in final_test
def generate_prediction_data(date_test, date_original, dataset_test, dataset_original, parameter):
    
    date = np.concatenate((date_original, date_test), axis=0)
    dataset_original_new = np.concatenate((dataset_original, dataset_test), axis=0)
    dataset = np.concatenate((dataset_original, dataset_test), axis=0)
    dataset = cal_moving_ave(dataset, parameter.moving_length)
#    dataset = reframeDF(dataset, scaler)

    return date, dataset_original_new, dataset


# Perform final model testing and evaluation
def final_test(dataset, parameter, model, criterion):
    
    # Prepare the prediction dataset by cutting time steps (no training, so is_train=False)
    fin_dataset, _ = cut_dataset_cycle(dataset, parameter.cycle_num, parameter.cycle_add, 
                                       parameter.day_num, parameter.train_pro, False, parameter.intervel)
    # Determine the number of full batches in the dataset
    num = int(fin_dataset.shape[0] / parameter.batch_size)
    # Calculate how many extra samples are needed to make a full batch
    plus = (num + 1) * parameter.batch_size - fin_dataset.shape[0]
    # Extend the dataset by repeating some samples to fit the batch size
    fin_dataset = torch.cat((fin_dataset, fin_dataset[:plus, :, :]), 0)
    print(fin_dataset.shape)  # Debugging: Check dataset shape after extension
    # Recalculate the number of batches after adjusting the dataset
    num = int(fin_dataset.shape[0] / parameter.batch_size)
    reminder = fin_dataset.shape[0] % parameter.batch_size  # Check if there are remaining samples
    print(reminder)  # Debugging: Remaining samples after batching
    print(num)  # Debugging: Updated number of batches

    # Lists to store final losses, predictions, and actual targets
    fin_losss, results, targets = [], [], []
    # Iterate over each batch
    for c in range(num):
        # Extract input features (sequence data) for the current batch
        input_x = fin_dataset[c * parameter.batch_size : c * parameter.batch_size + parameter.batch_size, 
                              :parameter.cycle_num + parameter.cycle_add, :]
        # Extract target values (ground truth for the last time step)
        target = fin_dataset[c * parameter.batch_size : c * parameter.batch_size + parameter.batch_size, 
                             parameter.cycle_num + parameter.cycle_add, 0]
        
        # Perform forward pass through the model to get predictions
        pred = model.forward(input_x).squeeze(1)

        # Store predictions and actual target values
        results.extend(pred.detach().numpy().tolist())
        targets.extend(target.tolist())
        
        # Compute the loss for this batch
        loss = criterion(pred, target)
        print(loss)  # Debugging: Print loss value
        fin_losss.append(loss.detach().numpy())  # Store loss value

    # Trim results and targets to match the original dataset size
    targets = targets[:fin_dataset.shape[0] - plus]
    results = results[:fin_dataset.shape[0] - plus]

    return results, targets


# Generate dataset information including average, minimum, and maximum values
def create_date_dic(dataset, parameter, date, dataset_original):
    
    # Generate a list of years based on the dataset's start year and total duration
    year_list = [str(i) for i in np.arange(parameter.start_year, parameter.start_year + parameter.total_year)]
    # Calculate the length of years covered in the dataset
    year_length = int(date[-1][:4]) - int(date[0][:4])

    # Dictionaries to store dataset values mapped to dates
    dataset_dic, dataset_wl = {}, {}
    total_num = len(date)
    # Populate dataset dictionaries with original dataset values
    for i in range(total_num):
        dataset_dic[date[i]] = dataset_original[i].tolist()  # Store full dataset entry
        dataset_wl[date[i]] = dataset_original[i, 0]  # Store only water level values

    # Define the number of days in each month (default values)
    month_length = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    init = 0  # Initialize index tracker for dataset entries
    month_average = {}  # Dictionary to store monthly average water levels
    # Loop through each year in the dataset
    for year_num in range(year_length + 1):
        year = 1981 + year_num  # Determine current year
        year_name = str(year)
        # Adjust February length for leap years
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0 and year % 3200 != 0):
            month_length[1] = 29
        else:
            month_length[1] = 28
        # Loop through each month
        for month in range(12):
            water_level_set = []  # Temporary list to store daily water levels
            # Format month as two-digit string
            month_name = str(month + 1) if month >= 9 else '0' + str(month + 1)
            # Collect water level values for each day in the month
            for i in range(month_length[month]):
                water_level_set.append(dataset_original[init][0])
                init += 1
            # Store the average water level for the month
            index = year_name + "-" + month_name
            month_average[index] = np.average(np.array(water_level_set))

    # Dictionaries to store the highest and lowest average monthly water levels
    max_dic, min_dic = {}, {}
    month_list = list(month_average.keys())
    # Loop through 12 months to find yearly min/max water levels
    for i in range(12):
        values = [month_average[month_list[i + j * 12]] for j in range(year_length + 1)]
        max_average = max(values)
        min_average = min(values)
        # Identify the year corresponding to the max/min water levels
        mark_max = str(parameter.start_year + values.index(max(values))) + '-' + str(i + 1)
        mark_min = str(parameter.start_year + values.index(min(values))) + '-' + str(i + 1)
        max_dic[mark_max] = max_average
        min_dic[mark_min] = min_average

    # Dictionaries to store daily information
    dataset_daily_info, dataset_daily_average, dataset_daily_max, dataset_daily_min = {}, {}, {}, {}
    # Initialize dictionary keys for daily records
    for j in date:
        if j[5:] not in dataset_daily_info:
            dataset_daily_info[j[5:]] = []
    # Sort dictionary keys for consistency
    dataset_daily_info = {key: dataset_daily_info[key] for key in sorted(dataset_daily_info.keys())}
    # Populate daily water level records
    for k in dataset_wl.keys():
        dataset_daily_info[k[5:]].append(dataset_wl[k])
    # Compute daily average water levels
    for m in dataset_daily_info.keys():
        dataset_daily_average[m] = sum(dataset_daily_info[m]) / len(dataset_daily_info[m])
    # Identify max/min water levels for each day across all years
    for themax in dataset_daily_info.keys():
        dataset_daily_max[themax] = [max(dataset_daily_info[themax]), 
                                     year_list[dataset_daily_info[themax].index(max(dataset_daily_info[themax]))]]
    for themin in dataset_daily_info.keys():
        dataset_daily_min[themin] = [min(dataset_daily_info[themin]), 
                                     year_list[dataset_daily_info[themin].index(min(dataset_daily_info[themin]))]]

    # Append daily average water level to the dataset dictionary
    for d in dataset_dic.keys():
        dataset_dic[d].append(dataset_daily_average[d[5:]])
    # Extract moving average values and concatenate with dataset
    dataset_with_ave = np.array(list(dataset_dic.values()))[parameter.moving_length:, 7]
    dataset_with_ave = dataset_with_ave[:, np.newaxis]
    dataset = np.concatenate((dataset, dataset_with_ave), axis=1)

    return dataset, dataset_daily_max, dataset_daily_min, max_dic, min_dic

# Analyze and plot prediction results on the testing dataset
def test_analysis(date, dataset, dataset_original, max_dic, min_dic, results, targets, parameter, scaler):
    
    # Extract the last `test_length` samples for evaluation
    results = np.array(results).reshape(-1, 1)[-parameter.test_length:]
    targets = np.array(targets).reshape(-1, 1)[-parameter.test_length:]
    date = np.array(date)[-parameter.test_length:]
    # Extract additional feature columns except the target variable
    others = dataset[-parameter.test_length:, 1:]
    # Recover original scale by applying inverse transformation
    recovery_preds = np.concatenate((results, others), axis=1)
    recovery_targets = np.concatenate((targets, others), axis=1)
    recovery_preds = scaler.inverse_transform(recovery_preds)
    recovery_targets = scaler.inverse_transform(recovery_targets)

    # Extract relevant data for plotting
    plot_preds = recovery_preds[:, 0].tolist()  # Predicted values
    plot_targets = recovery_targets[:, 0].tolist()  # Actual values (ground truth)
    plot_original = dataset_original[-parameter.test_length:, 0].tolist()  # Original dataset values
    plot_average = recovery_preds[:, 7].tolist()  # Long-term average for comparison

    # Compute RMSE error
    plot_preds = np.array(plot_preds)
    plot_targets = np.array(plot_targets)
    mse = np.mean(np.square(plot_preds - plot_targets))
    error = np.sqrt(mse)

    # Create figure for visualization
    fig, ax1 = plt.subplots(figsize=(20, 10))
    # Configure major x-axis locator for better time visualization
    x_major_locator = MultipleLocator(30)
    # Create secondary y-axis for unit conversion (Meters to Feet)
    ax2 = ax1.twinx()
    ax1.grid()

    years_long, init = [], 0  
    # Loop through predicted years and adjust for leap years
    for i in range(parameter.predict_long):
        year = parameter.predict_year_start + 1
        the_last_day = ['01-31', '02-28', '03-31', '04-30', '05-31', '06-30' ,'07-31', '08-31', '09-30', '10-31', '11-30', '12-31']
        # Adjust February length for leap years
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0 and year % 3200 != 0):
            the_last_day[1] = '02-29'

        # Extract max and min water levels for each month
        max_dic_new, min_dic_new = {}, {}
        max_index_list, max_values_list = list(max_dic.keys()), list(max_dic.values())
        min_index_list, min_values_list = list(min_dic.keys()), list(min_dic.values())
        for i in range(12):
            max_dic_new[i] = [max_index_list[i][:4], max_values_list[i]]
            min_dic_new[i] = [min_index_list[i][:4], min_values_list[i]]

        # Plot monthly max and min water levels
        for mon in range(12):
            p_x = date[init:init + int(the_last_day[mon][3:])]
            init += int(the_last_day[mon][3:])
            y1 = [max_dic_new[mon][1] for _ in p_x]
            y2 = [min_dic_new[mon][1] for _ in p_x]

            ax1.plot(p_x, y1, c='r')  # Max values
            ax2.plot(p_x, [i * 3.281 for i in y1], c='r')  # Convert to feet
            ax1.plot(p_x, y2, c='g')  # Min values
            ax2.plot(p_x, [i * 3.281 for i in y2], c='g')  # Convert to feet

            ax1.text(p_x[2], y1[0] + 0.01, max_dic_new[mon][0], size=10, weight='normal')
            ax1.text(p_x[2], y2[0] - 0.05, min_dic_new[mon][0], size=10, weight='normal')

    # Plot predicted results and corresponding actual values
    ax1.plot(date, plot_preds, label='Predicted results')
    ax1.xaxis.set_major_locator(x_major_locator)  # Set major x-axis locator
    ax2.plot(date, [i * 3.281 for i in plot_preds])  # Convert predictions to feet
    ax1.plot(date, plot_original, label='Original data')
    ax2.plot(date, [j * 3.281 for j in plot_original])  # Convert original data to feet
    ax1.plot(date, plot_targets, label='Moving-averaged data') 
    ax2.plot(date, [k * 3.281 for k in plot_targets])  # Convert moving-averaged data to feet
    ax1.plot(date, plot_average, '--', label='Long-term average') 
    ax2.plot(date, [k * 3.281 for k in plot_average], '--')  # Convert long-term average to feet

    # Set labels for both y-axes
    ax1.set_xlabel('Time', size=15)
    ax1.set_ylabel('Meters', size=15)
    ax2.set_ylabel('Feet', size=15)

    # Set limits for better visualization
    ax1.set_ylim(182.6, 184.2)
    ax2.set_ylim(182.6 * 3.281, 184.2 * 3.281)

    # Display legend and title
    ax1.legend(loc='best')
    plt.title('Prediction of Water Level on Test Dataset', size=18)
    
    # Auto-format date labels for better readability
    plt.gcf().autofmt_xdate()
    
    # Show plot
    plt.show()

    return error, plot_targets, plot_preds