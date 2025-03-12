import torch

# Configuration class for the auto-weighted model, defining layer sizes
class auto_weighted_config():
    def __init__(self):
        self.transform_layer_1 = 128  # Number of neurons in the first transformation layer
        self.integrate_1 = 128  # Number of neurons in the first integration layer
        self.integrate_2 = 256  # Number of neurons in the second integration layer
        self.integrate_3 = 16   # Number of neurons in the third integration layer
        self.weighted_1 = 512   # Number of neurons in the first weighted layer
        self.weighted_2 = 128   # Number of neurons in the second weighted layer
        self.weighted_3 = 16    # Number of neurons in the third weighted layer
        self.output_size = 2    # Output size (number of weights)

# Instantiate the configuration class
awc = auto_weighted_config()


# Auto-weighted model using multiple linear layers and softmax
class auto_weighted(torch.nn.Module):
    def __init__(self, parameter):
        super(auto_weighted, self).__init__()

        # Extract input parameters from the given config object
        self.feature_size = parameter.input_size  # Number of input features
        self.cycle_num_before = parameter.cycle_num_before  # Number of time steps in the history
        self.cycle_num_later = parameter.cycle_num_later  # Number of future time steps

        # Transformation module: Processes historical data
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size, awc.transform_layer_1, bias=False),  # First linear layer
            torch.nn.ReLU(),  # Activation function
            torch.nn.Linear(awc.transform_layer_1, self.feature_size - 1, bias=False)  # Second linear layer
        )

        # Integration module: Combines transformed historical data with future data
        self.integrate = torch.nn.Sequential(
            torch.nn.Linear(self.feature_size - 1, awc.integrate_1, bias=False),  # First integration layer
            torch.nn.ReLU(),  # Activation function
            torch.nn.Linear(awc.integrate_1, awc.integrate_2, bias=False),  # Second integration layer
            torch.nn.ReLU(),  # Activation function
            torch.nn.Linear(awc.integrate_2, awc.integrate_3, bias=False)  # Third integration layer
        )

        # Weight computation module: Produces weights for prediction
        self.weighted = torch.nn.Sequential(
            torch.nn.Linear((self.cycle_num_before + self.cycle_num_later) * awc.integrate_3, awc.weighted_1, bias=False),  # First weighted layer
            torch.nn.ReLU(),  # Activation function
            torch.nn.Linear(awc.weighted_1, awc.weighted_2, bias=False),  # Second weighted layer
            torch.nn.ReLU(),  # Activation function
            torch.nn.Linear(awc.weighted_2, awc.weighted_3, bias=False),  # Third weighted layer
            torch.nn.ReLU(),  # Activation function
            torch.nn.Linear(awc.weighted_3, awc.output_size, bias=False)  # Output layer
        )

    def forward(self, dataset):
        # Get the batch size
        dataset_size = dataset.shape[0]

        # Extract historical and future data from the input dataset
        dataset_history = dataset[:, :self.cycle_num_before, :]  # Historical sequence
        dataset_future = dataset[:, self.cycle_num_before:self.cycle_num_before + self.cycle_num_later, 1:]  # Future sequence

        # Apply transformation to historical data
        history_transform = self.transform(dataset_history)

        # Concatenate transformed historical data with future data
        dataset_integrate = torch.cat([history_transform, dataset_future], 1)

        # Apply integration layers
        dataset_integrate = self.integrate(dataset_integrate)

        # Flatten integrated data for weight computation
        dataset_to_weight = dataset_integrate.view(dataset_size, -1)

        # Compute final weights using softmax activation
        weights = torch.softmax(self.weighted(dataset_to_weight), 1)

        return weights