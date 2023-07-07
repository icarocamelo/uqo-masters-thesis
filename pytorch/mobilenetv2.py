import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def train():
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     # Your training loop here

        # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

        # Define transforms for data augmentation
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
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                shuffle=False, num_workers=2)

    # Define MobileNetV2 model
    model = torchvision.models.mobilenet_v2(pretrained=False)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training the model
    num_epochs = 1
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for epoch in range(num_epochs):
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

                break

    print('Finished training')

    # Evaluate the model
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy on the test set: %.2f %%' % accuracy)

if __name__ == '__main__':
    train()
