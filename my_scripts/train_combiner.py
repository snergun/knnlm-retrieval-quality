import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from data_structures import Dataset, Dstore

# Define the neural network
class FeedForwardNN(nn.Module):
    def __init__(self, input_size=50, hidden_size=100, output_size=3):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, predictions):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        out = (x * predictions).sum(-1)
        return out

# Dataset class for your custom data
class CustomDataset(TorchDataset):
    def __init__(self, features, values, labels):
        self.features = features
        self.values = predictions
        self.labels = labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.values[idx], self.labels[idx]

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print statistics every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
def load_dataset(pos_probs_path, knn_probs_path, labels_path):
    # Local variables.
    dstore = context['dstore']
    keys = dstore.keys
    vals = dstore.vals
    dataset = context['dataset']
    query = dataset.query
    target = dataset.target
    knns = dataset.knns
    dists = context['dists']
    #External LM Probabilities
    ext_lm_prob = context.get('ext_lm_prob')
    ext_lm_modified_prob = context.get('ext_lm_modified_prob')
    ext_weight = context.get('ext_weight')
    # LM perplexity.
    log_progress("get kNN probabilities")
    knn_prob = get_knn_prob(dstore, target, dists, knns).view(-1, 1)
    log_progress("kNN probabilities calculated")
    lm_prob = torch.from_numpy(dataset.prob).float()
    return pos_probs, knn_probs, labels
# Example usage
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create some dummy data (replace with your actual data)
    # X should be of shape (num_samples, 50)
    # y should be of shape (num_samples, 3) for one-hot encoded labels
    # or (num_samples,) for class indices
    
    # Load your data here
    # For example, if your data is in a CSV file:
    # import pandas as pd
    # data = pd.read_csv('your_data.csv')
    # X = data.iloc[:, :-1].values  # Assuming the last column is the label
    # y = data.iloc[:, -1].values   # The last column as labels
    
    # For this example, creating dummy data
    num_samples = 1000
    X = torch.randn(num_samples, 50)
    
    # Creating random labels (0, 1, or 2)
    y_indices = torch.randint(0, 3, (num_samples,))
    
    # Convert to one-hot encoding
    y = torch.zeros(num_samples, 3)
    y.scatter_(1, y_indices.unsqueeze(1), 1)
    
    # Create dataset and dataloader
    dataset = CustomDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize the model
    model = FeedForwardNN(input_size=50, hidden_size=100, output_size=3)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=100)
    
    # Save the trained model
    torch.save(model.state_dict(), 'ffnn_model.pth')
    
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()