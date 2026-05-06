# Install dependencies first:
# pip install flwr torch torchvision numpy

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# -------------------------------
# Define CNN model
# -------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# -------------------------------
# Training with Differential Privacy
# -------------------------------
def train(model, train_loader, optimizer, epochs=1, noise_scale=0.01):
    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # Inject Gaussian noise into gradients for privacy
            for param in model.parameters():
                param.grad += noise_scale * torch.randn_like(param.grad)

            optimizer.step()

def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(test_loader.dataset)

# -------------------------------
# Federated Client
# -------------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader, client_id):
        self.model = Net()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        train(self.model, self.train_loader, optimizer, epochs=2, noise_scale=0.05)
        print(f"Client {self.client_id} finished training")
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = test(self.model, self.test_loader)
        print(f"Client {self.client_id} accuracy: {accuracy:.4f}")
        return float(1.0 - accuracy), len(self.test_loader.dataset), {"accuracy": accuracy}

# -------------------------------
# Data loading with heterogeneous splits
# -------------------------------
def load_data(client_id, num_clients=3):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    # Non-IID split: each client gets digits biased towards certain labels
    labels = np.array(train_dataset.targets)
    client_labels = labels % num_clients
    mask = client_labels == client_id
    client_data = torch.utils.data.Subset(train_dataset, np.where(mask)[0])

    train_loader = torch.utils.data.DataLoader(client_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    return train_loader, test_loader

# -------------------------------
# Custom Aggregation Strategy
# -------------------------------
class WeightedAverage(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # Weighted averaging based on dataset size
        weights = []
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        for _, fit_res in results:
            weight = fit_res.num_examples / total_samples
            weights.append(weight)
        print(f"Round {rnd}: Aggregated with weighted averaging")
        return super().aggregate_fit(rnd, results, failures)

# -------------------------------
# Run server or client
# -------------------------------
def run_server():
    strategy = WeightedAverage()
    fl.server.start_server(server_address="127.0.0.1:8080", strategy=strategy, config={"num_rounds": 3})

def run_client(client_id):
    train_loader, test_loader = load_data(client_id)
    client = FlowerClient(train_loader, test_loader, client_id)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        run_server()
    else:
        client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        run_client(client_id)
