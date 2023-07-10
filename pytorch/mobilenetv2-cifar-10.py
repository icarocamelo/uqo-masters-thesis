import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.profiler import profile, record_function, ProfilerActivity
import warnings


# Disabling warnings
warnings.filterwarnings("ignore")


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

# Define MobileNetV2 model
mobilenet = torchvision.models.mobilenet_v2(pretrained=False)
num_features = mobilenet.classifier[1].in_features
mobilenet.classifier[1] = nn.Linear(num_features, 10)  # Modify the last fully connected layer for 10 classes

# Move the model to the device (GPU/CPU)
mobilenet = mobilenet.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mobilenet.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = mobilenet(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 200 == 199:  # Print average loss every 200 mini-batches
            print(f'Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished training')

print('Calculating metrics...')
# Evaluation on test set
true_labels = []
predicted_labels = []

with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        with record_function("Model Inference"):
            outputs = mobilenet(images)
            _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)) 
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


# Convert lists to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Calculate precision, recall, and F-score
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f_score = f1_score(true_labels, predicted_labels, average='macro')

print("Precision:", precision)
print("Recall:", recall)
print("F-score:", f_score)

