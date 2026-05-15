import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# ==========================================
# IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()
])

# ==========================================
# LOAD TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(

    root="split_dataset/test",
    transform=transform
)

test_loader = DataLoader(

    test_dataset,
    batch_size=16,
    shuffle=False
)

classes = test_dataset.classes

print("Classes:", classes)

# ==========================================
# CNN MODEL
# ==========================================

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

# ==========================================
# LOAD BEST MODEL
# ==========================================

model = RoadCNN().to(device)

model.load_state_dict(torch.load("best_road_model.pth"))

model.eval()

print("\nBest Model Loaded Successfully!")

# ==========================================
# PREDICTIONS
# ==========================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())

        all_predictions.extend(predicted.cpu().numpy())

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

report_dict = classification_report(

    all_labels,
    all_predictions,
    target_names=classes,
    output_dict=True
)

report = classification_report(

    all_labels,
    all_predictions,
    target_names=classes
)

print("\nClassification Report:\n")

print(report)

# ==========================================
# CONVERT TO DATAFRAME
# ==========================================

df = pd.DataFrame(report_dict).transpose()

# Round values

df = df.round(2)

# ==========================================
# SAVE REPORT AS PNG
# ==========================================

fig, ax = plt.subplots(figsize=(12, 5))

ax.axis('off')

table = ax.table(

    cellText=df.values,
    colLabels=df.columns,
    rowLabels=df.index,
    cellLoc='center',
    loc='center'
)

# ==========================================
# TABLE FORMATTING
# ==========================================

table.auto_set_font_size(False)

table.set_fontsize(11)

table.scale(1.3, 2)

# ==========================================
# TITLE
# ==========================================

plt.title(

    "Classification Report",
    fontsize=16,
    fontweight='bold',
    pad=20
)

# ==========================================
# SAVE IMAGE
# ==========================================

plt.tight_layout()

plt.savefig(

    "classification_report.png",
    bbox_inches='tight',
    dpi=300
)

plt.show()

print("\nClassification report image saved successfully!")