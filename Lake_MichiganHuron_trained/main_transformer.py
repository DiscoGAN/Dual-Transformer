import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from garage_Transformer import cal_moving_ave, reframeDF, create_date_dic, final_test
from garage_Transformer import final_test, test_analysis, generate_prediction_data
from model_base import Transformer_half_base
from model_modify import Transformer_half_modify
from model_weighted import auto_weighted
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# Configuration class for model training parameters
class config():
    def __init__(self):
        self.cycle_num_before = 12  # Number of historical time steps
        self.cycle_num_later = 5  # Number of future time steps
        self.moving_length = 30  # Moving average window size
        self.interval_1 = 30  # Interval for historical input dataset
        self.interval_2 = 30  # Interval for future input dataset
        self.is_mean = True  # Whether to use moving-average operation
        self.is_train = True  # Flag to indicate training mode
        self.input_size = 8  # Number of input features (historical)
        self.batch_size = 256  # Batch size for training
        self.output_size = 1 # Output size
        self.train_pro = 1 # Training dataset porpotion
        
        # Model file paths
        self.path_base = './model/transformer_base_6.pt'
        self.path_modify = './model/transformer_modify_6.pt'
        self.path_weight = './model/transformer_aw_6.pt'
        
        # Dataset paths
        self.dataset_path = './dataset/dataset_whole.npy'
        self.date_path = './dataset/date.npy'
        
        self.test_length = 365 * 2  # Length of the test dataset (2 years)
        self.start_year = 1981  # Starting year for dataset
        self.total_year = 100  # Total available years in the dataset (fake)
        self.predict_year_start = 2022  # Start year for predictions
        self.plot_start_year = self.predict_year_start - 1  # Year to start visualization
        self.predict_long = 2  # Number of years to predict
        self.base = 0.0006  # Optimization criterion

# Initialize configuration
wl_p = config()

# Load dataset and date information
date_whole = list(np.load(wl_p.date_path))
dataset_whole = np.load(wl_p.dataset_path)
scaler = MinMaxScaler(feature_range=(0, 1))

# Split dataset into original and test parts
date_original, date = date_whole[:-wl_p.test_length], date_whole[:-wl_p.test_length]
dataset_original, dataset = dataset_whole[:-wl_p.test_length], dataset_whole[:-wl_p.test_length]

# Apply moving average filtering
dataset = cal_moving_ave(dataset, wl_p.moving_length)
print(dataset.shape)

# Normalize dataset
dataset = reframeDF(dataset, scaler)

# Load test dataset
date_test, dataset_test = date_whole[-wl_p.test_length:], dataset_whole[-wl_p.test_length:]

# Initialize models
transformer_base = Transformer_half_base(wl_p)
transformer_modify = Transformer_half_modify(wl_p)
aw = auto_weighted(wl_p)
criterion = torch.nn.MSELoss()  # Loss function

# Load pre-trained models
transformer_base.load_state_dict(torch.load(wl_p.path_base))
transformer_base.eval()

transformer_modify.load_state_dict(torch.load(wl_p.path_modify))
transformer_modify.eval()

aw.load_state_dict(torch.load(wl_p.path_weight))
aw.eval()

# Switch to evaluation mode
wl_p.is_train = False

# Generate dataset for prediction
date, dataset_original, dataset_new = generate_prediction_data(date_test, date_original, dataset_test, dataset_original, wl_p)

# Create dataset dictionary with min/max/average values
dataset_new, dataset_daily_max, dataset_daily_min, max_dic, min_dic = create_date_dic(dataset_new, wl_p, date, dataset_original)

# Normalize the dataset for prediction
dataset_new = reframeDF(dataset_new, scaler)

# Perform final test using trained models
results, targets, weights_1, weights_2 = final_test(dataset_new, wl_p, transformer_base, transformer_modify, aw, criterion)
np.save('./results/results.npy', results)
np.save('./results/targets.npy', targets)

results = np.load('./results/results.npy')
targets = np.load('./results/targets.npy')

# Analyze test results
error, error_pearson, plot_targets, plot_preds = test_analysis(date, dataset_new, dataset_original, max_dic, min_dic, results, targets, wl_p, scaler)
print(len(plot_targets), len(plot_preds))

# Compute mean absolute error
print(np.mean((abs(np.array(plot_targets) - np.array(plot_preds)))))

# Compute RMSE
test_error = np.array(plot_targets) - np.array(plot_preds)
test_rmse = np.sqrt(sum(np.square(test_error)) / wl_p.test_length)
print(test_rmse)

# Compute correlation coefficient
pearson_corr, _ = pearsonr(plot_targets, plot_preds)
print(pearson_corr)

# Compute R-squared score
r2 = r2_score(plot_targets, plot_preds)
print(r2)