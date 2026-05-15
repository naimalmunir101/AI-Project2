import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(brightness=0.2),

    transforms.ToTensor()
])

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()
])


train_dataset = datasets.ImageFolder(

    root="split_dataset/train",
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(

    root="split_dataset/valid",
    transform=test_transform
)

train_loader = DataLoader(

    train_dataset,
    batch_size=16,
    shuffle=True
)

valid_loader = DataLoader(

    valid_dataset,
    batch_size=16,
    shuffle=False
)

classes = train_dataset.classes

print("Classes:", classes)

class RoadCNN(nn.Module):

    def __init__(self):

        super(RoadCNN, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 28 * 28, 512),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(512, 4)
        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x


model = RoadCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(

    model.parameters(),
    lr=0.001
)

train_losses = []

train_accuracies = []

val_accuracies = []

best_val_accuracy = 0

epochs = 15

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    average_loss = running_loss / len(train_loader)


    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    train_losses.append(average_loss)

    train_accuracies.append(train_accuracy)

    val_accuracies.append(val_accuracy)

    print(f"\nEpoch [{epoch+1}/{epochs}]")

    print(f"Training Loss: {average_loss:.4f}")

    print(f"Training Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    print("-" * 40)

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(model.state_dict(), "best_road_model.pth")

        print("Best Model Saved!")

torch.save(model.state_dict(), "road_model.pth")

print("\nFinal Model Saved Successfully!")

epochs_range = range(1, epochs + 1)

#Accuracy graph
plt.figure(figsize=(8,5))

plt.plot(epochs_range, train_accuracies, label='Training Accuracy')

plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.title('Training vs Validation Accuracy')

plt.legend()

plt.grid(True)

plt.savefig("accuracy_graph.png")

plt.show()

#loss graph
plt.figure(figsize=(8,5))

plt.plot(epochs_range, train_losses, label='Training Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Training Loss Graph')

plt.legend()

plt.grid(True)

plt.savefig("loss_graph.png")

plt.show()