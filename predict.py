import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image

# ==========================================
# DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# CLASS NAMES
# ==========================================

classes = ['Crack', 'Normal', 'Patchwork', 'Pothole']

# ==========================================
# IMAGE TRANSFORM
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()
])

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
# LOAD MODEL
# ==========================================

model = RoadCNN().to(device)

model.load_state_dict(torch.load("best_road_model.pth"))

model.eval()

print("Best Model Loaded Successfully!")

# ==========================================
# IMAGE PATH
# ==========================================

image_path = "test.jpg"

# ==========================================
# LOAD IMAGE
# ==========================================

image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)

# ==========================================
# PREDICTION
# ==========================================

with torch.no_grad():

    outputs = model(image)

    _, predicted = torch.max(outputs, 1)

predicted_class = classes[predicted.item()]

print("\nPredicted Class:", predicted_class)