import torch
import torch.nn as nn
# Define the neural network for combining probabilities
# Loss function for perplexity
class PerplexityLoss(nn.Module):
    def __init__(self):
        super(PerplexityLoss, self).__init__()
    
    def forward(self, weights, probs, targets):
        """
        weights: Model output weights for each distribution [batch_size, num_probs]
        probs: List of probability distributions [lm_probs, knn_probs, pos_modified_probs]
                each with shape [batch_size]
        targets: Target tokens [batch_size]
        """
        # Stack probabilities
        stacked_probs = torch.stack(probs, dim=1)  # [batch_size, num_probs]
        
        # Apply weights (in log space)
        combined_log_probs = torch.logsumexp(torch.log(weights) + stacked_probs, dim=1)
        
        # Calculate negative log likelihood (equivalent to perplexity)
        return -combined_log_probs.mean()
class ProbCombiner(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=100, num_probs=3, dropout_rate=0.2, input_dropout_rate=0.1):
        super(ProbCombiner, self).__init__()
        # Optional input dropout
        self.input_dropout = nn.Dropout(input_dropout_rate)
        
        # First layer
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Second layer
        self.fc2 = nn.Linear(hidden_dim, num_probs)
        self.layer_norm2 = nn.LayerNorm(num_probs)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, features):
        # Optional input dropout
        x = self.input_dropout(features)
        
        # First layer with normalization
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second layer with normalization
        x = self.fc2(x)
        x = self.layer_norm2(x)
        
        # Output weights that sum to 1 using softmax
        weights = self.softmax(x)
        return weights