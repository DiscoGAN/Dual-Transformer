import torch
import numpy as np

# Configuration class for Transformer model
class Transconfig():
    def __init__(self):
        self.d_model = 7      # Dimensionality of the model
        self.d_ff = 4096      # Hidden layer size in feed-forward networks
        self.d_k = self.d_v = 64  # Dimensions of key and value vectors
        self.n_layers = 8     # Number of transformer layers
        self.n_heads = 12     # Number of attention heads
        self.fc_p = 128       # Fully connected layer size
        self.output_size = 1  # Output size
        self.batch_size = 256 # Batch size for training
        self.cycle_num = 17   # Number of cycles (sequence length)

transconfig = Transconfig()


# Positional Encoding to add positional information to input embeddings
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model=transconfig.d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Create position encoding table
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)
        ])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # Apply sine to even indices
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # Apply cosine to odd indices
        self.pos_table = torch.FloatTensor(pos_table)  # Convert to PyTorch tensor

    def forward(self, enc_inputs):
        # Add positional encoding to input embeddings
        enc_inputs = enc_inputs + self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)


# Scaled Dot-Product Attention mechanism
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(transconfig.d_k)
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


# Multi-Head Attention layer
class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = torch.nn.Linear(transconfig.d_model, transconfig.d_k * transconfig.n_heads, bias=False)
        self.W_K = torch.nn.Linear(transconfig.d_model, transconfig.d_k * transconfig.n_heads, bias=False)
        self.W_V = torch.nn.Linear(transconfig.d_model, transconfig.d_v * transconfig.n_heads, bias=False)
        self.fc = torch.nn.Linear(transconfig.n_heads * transconfig.d_v, transconfig.d_model, bias=False)
        self.dropout = torch.nn.Dropout(p=0.1)  # Dropout to prevent overfitting

    def forward(self, input_Q, input_K, input_V):
        # Get batch size
        residual, batch_size = input_Q, input_Q.size(0)

        # Compute Q, K, V matrices for multi-head attention
        Q = self.W_Q(input_Q).view(batch_size, -1, transconfig.n_heads, transconfig.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, transconfig.n_heads, transconfig.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, transconfig.n_heads, transconfig.d_v).transpose(1, 2)

        # Apply scaled dot-product attention
        context, attn = ScaledDotProductAttention()(Q, K, V)

        # Concatenate heads and apply final linear transformation
        context = context.transpose(1, 2).reshape(batch_size, -1, transconfig.n_heads * transconfig.d_v)
        output = self.fc(context)
        output = self.dropout(output)  # Apply dropout

        # Add residual connection and layer normalization
        return torch.nn.LayerNorm(transconfig.d_model)(output + residual), attn


# Position-wise Feed-Forward Network
class PoswiseFeedForwardNet(torch.nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(transconfig.d_model, transconfig.d_ff, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(transconfig.d_ff, transconfig.d_model, bias=False)
        )

    def forward(self, inputs):
        # Apply feed-forward network with residual connection
        residual = inputs
        output = self.fc(inputs)
        return torch.nn.LayerNorm(transconfig.d_model)(output + residual)


# Transformer Encoder Layer (Multi-Head Attention + Feed-Forward Network)
class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # Multi-head self-attention
        self.pos_ffn = PoswiseFeedForwardNet()     # Feed-forward network

    def forward(self, enc_inputs):
        # Apply self-attention and feed-forward network
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


# Transformer Encoder with multiple layers
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(transconfig.d_model)  # Positional encoding
        self.layers = torch.nn.ModuleList([EncoderLayer() for _ in range(transconfig.n_layers)])

    def forward(self, enc_inputs):
        # Apply positional encoding
        enc_outputs = self.pos_emb(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            # Pass through multiple encoder layers
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


# Double Encoder for enhanced feature extraction
class DoubleEncoder(torch.nn.Module):
    def __init__(self):
        super(DoubleEncoder, self).__init__()
        self.encoder1 = Encoder()  # First encoder
        # self.encoder2 = Encoder()  # Uncomment to add a second encoder

    def forward(self, enc_inputs):
        # Apply first encoder
        enc_outputs1, enc_self_attns1 = self.encoder1(enc_inputs)
        # enc_outputs2, enc_self_attns2 = self.encoder2(enc_outputs1)  # Uncomment for second encoder
        return enc_outputs1, enc_self_attns1


# Prediction layer for final output
class PredictLayer(torch.nn.Module):
    def __init__(self):
        super(PredictLayer, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(transconfig.cycle_num * transconfig.d_model, transconfig.fc_p, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(transconfig.fc_p, transconfig.output_size, bias=False)
        )

    def forward(self, enc_outputs):
        # Flatten input and apply fully connected layers
        enc_outputs = enc_outputs.reshape(-1, transconfig.cycle_num * transconfig.d_model)
        output = self.fc(enc_outputs)
        return output


# Full Transformer model
class Transformer(torch.nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.DoubleEncoder = DoubleEncoder()  # Use double encoder for feature extraction
        self.Predict = PredictLayer()         # Prediction layer

    def forward(self, enc_inputs):
        # Pass input through encoder and prediction layer
        enc_outputs, enc_self_attns1 = self.DoubleEncoder(enc_inputs)
        outputs = self.Predict(enc_outputs)
        return outputs