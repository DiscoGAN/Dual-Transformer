import torch

# Define a configuration class for the auto-weighted model
class auto_weighted_config():
    def __init__(self):
        self.transform_layer_1 = 128  # First layer of transformation block
        self.integrate_1 = 128        # First layer of integration block
        self.integrate_2 = 256        # Second layer of integration block
        self.integrate_3 = 16         # Third layer of integration block
        self.weighted_1 = 512         # First layer of weighting block
        self.weighted_2 = 128         # Second layer of weighting block
        self.weighted_3 = 16          # Third layer of weighting block
        self.output_size = 2          # Output size (number of weights)

# Initialize configuration
awc = auto_weighted_config()


# Define the auto-weighted module
class auto_weighted(torch.nn.Module):
    def __init__(self, parameter):
        super(auto_weighted, self).__init__()

        # Model hyperparameters
        self.feature_size = parameter.input_size          # Number of input features
        self.cycle_num_before = parameter.cycle_num_before  # Number of past cycles
        self.cycle_num_later = parameter.cycle_num_later    # Number of future cycles

        # Transformation layer: reduces feature size
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, awc.transform_layer_1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.transform_layer_1, self.feature_size - 1, bias=False)
        )

        # Integration layer: processes transformed historical and future data
        self.integrate = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size - 1, awc.integrate_1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.integrate_1, awc.integrate_2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.integrate_2, awc.integrate_3, bias=False)
        )

        # Weighting layer: assigns dynamic weights to historical and future data
        self.weighted = torch.nn.Sequential(
            torch.nn.Linear((self.cycle_num_before + self.cycle_num_later) * awc.integrate_3, awc.weighted_1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.weighted_1, awc.weighted_2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.weighted_2, awc.weighted_3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(awc.weighted_3, awc.output_size, bias=False)
        )

    def forward(self, dataset):
        dataset_size = dataset.shape[0]

        # Split dataset into historical and future components
        dataset_history = dataset[:, :self.cycle_num_before, :]  # Past data
        dataset_future = dataset[:, self.cycle_num_before:self.cycle_num_before + self.cycle_num_later, 1:]  # Future data (excluding first feature)

        # Apply transformation to historical data
        history_transform = self.transform(dataset_history)
        # Concatenate transformed historical data with future data
        dataset_integrate = torch.cat([history_transform, dataset_future], 1)
        # Apply integration layer
        dataset_integrate = self.integrate(dataset_integrate)
        # Flatten the dataset for the weighting layer
        dataset_to_weight = dataset_integrate.view(dataset_size, -1)
        # Compute weights using softmax
        weights = torch.softmax(self.weighted(dataset_to_weight), 1)

        return weights