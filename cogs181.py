import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import timm
print(timm.__version__)  # Check if it's correctly installed

# Define device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data Preparation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomCrop(32, padding=4),  # More data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# Load CIFAR-10 dataset
batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Define CNN Model (ResNet-18)
import torchvision.models as models

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)
model = model.to(device)


# 3. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

# 4. Training Function
def train(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    return train_losses

# 5. Testing Function
def test(model, testloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# 6. Run Training and Testing
import datetime

if __name__ == '__main__':
    # Define number of epochs
    epochs = 20  # Adjust as needed

    # Run training
    train_losses = train(model, trainloader, criterion, optimizer, epochs)

    # Run testing
    test_accuracy = test(model, testloader)

    # Save results to log file
    log_filename = "training_log.txt"

    with open(log_filename, "a") as f:
        f.write(f"--- Training Run ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
        # f.write(f"Model: ResNet-18\n")  # Change if using ResNet-34
        f.write(f"Model: {model.__class__.__name__}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"Optimizer: {optimizer.__class__.__name__}\n")
        f.write(f"Final Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        f.write("-" * 50 + "\n")

    print(f"Results saved to {log_filename}")

    # Plot Training Loss Curve (move inside this block)
    import matplotlib.pyplot as plt
    plt.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), "cnn_cifar10.pth")
    print("Model saved successfully!")

    

# import timm

# # Define ViT model
# model_vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)  # Adjust for CIFAR-10
# model_vit = model_vit.to(device)

# transform_vit = transforms.Compose([
#     transforms.Resize(224),  # ViT needs 224x224 input
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# trainset_vit = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_vit)
# testset_vit = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_vit)

# trainloader_vit = torch.utils.data.DataLoader(trainset_vit, batch_size=64, shuffle=True, num_workers=2)
# testloader_vit = torch.utils.data.DataLoader(testset_vit, batch_size=64, shuffle=False, num_workers=2)

# optimizer_vit = torch.optim.AdamW(model_vit.parameters(), lr=3e-4, weight_decay=1e-2)
