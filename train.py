import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("Script started")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import Net
from utils import sparsity_loss, evaluate, calculate_sparsity, plot_gate_distribution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)


def train_model(lambda_val):
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            ce_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model)

            loss = ce_loss + lambda_val * sp_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model


if __name__ == "__main__":
    lambdas = [0.001, 0.01, 0.1]

    for lam in lambdas:
        print(f"\nTraining with lambda = {lam}")
        model = train_model(lam)

        acc = evaluate(model, testloader, device)
        sparsity = calculate_sparsity(model)

        print(f"Lambda: {lam}, Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")

    # Plot for last model
    plot_gate_distribution(model)