# ROAD DAMAGE DETECTION SYSTEM 🛣️

---

## **📌 Project Overview**

Hello and welcome! 

The **main purpose** of this project is to automatically detect different types of road conditions using Deep Learning and Computer Vision techniques. Road damage is a very common problem in many countries. Cracks, potholes and damaged roads can cause:

- traffic accidents
- vehicle damage
- uncomfortable travel
- maintenance issues

Traditionally, road inspection is done manually which is time-consuming, expensive and unsafe in some situations. To solve this problem, we developed a CNN-based image classification system using PyTorch that can classify road images into **four categories**:

- Crack
- Normal Road
- Patchwork
- Pothole

---

## **👥 Team Information**

- **Group Number:** 1
- **Team Members:**
  - **Naimal Munir**
  - **Eman Fatima**
  - **Fatima Muaz**

---

## **💡 Project Objectives**

The main objectives of our project are:

- To build an automated road damage detection system
- To classify road images into multiple categories
- To use Convolutional Neural Networks (CNN) for image classification
- To train the model using PyTorch
- To improve understanding of Deep Learning concepts and Computer Vision
  
---

## **🛠️ Technologies Used**

In this project, we used the following technologies:
<img width="374" height="230" alt="image" src="https://github.com/user-attachments/assets/87b42d46-88e8-4dbf-bdae-209ed147475d" />

---

## **📂 Dataset**

For this project, we collected our own custom dataset using mobile phone camera images and raw internet images

### Dataset Size
- Total Images: 428
- Images per class: 107

### Splitting

We divided the dataset into:
<img width="335" height="115" alt="image" src="https://github.com/user-attachments/assets/c0f65c23-48bb-4df2-aae7-2ca378212f2c" />

This helps the model:

- learn properly
- validate performance
- test on unseen images

---

 ## **🖼️ Image Preprocessing**

Before training the CNN model, we applied preprocessing techniques such as:

- Image resizing (224 × 224)
- Image normalization
- Horizontal flipping
- Rotation augmentation
- Brightness adjustment

These techniques help improve model generalization and reduce overfitting.

---

## **🧠 CNN Model Architecture**

We implemented a custom **CNN model using PyTorch**.

The architecture includes:
- Convolution Layers
- ReLU Activation Functions
- Max Pooling Layers
- Fully Connected Layers
- Dropout Layer

The CNN automatically learns textures, shapes, road patterns and damage features from the input images.

**1- Model Training**

The model was trained using:
- Loss Function: CrossEntropyLoss()
-  Optimizer: Adam Optimizer
-  Epochs: 15

**2- Model Checkpointing**

The system automatically saves the best model based on validation accuracy.

**3- Model Evaluation**

To evaluate our model, we used:

- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Confusion Matrix
- Classification Report
- Accuracy Graph
- Loss Graph

---

## **📊 Final Results**

<img width="448" height="144" alt="image" src="https://github.com/user-attachments/assets/168a48c8-ef1d-430a-b4f1-0d1f6abb1f14" />

The model successfully learned road damage patterns and produced good classification performance considering the custom dataset size.

---

## **📈 Model Performance Graphs**

**1- Training vs Validation Accuracy Graph**

This graph shows how the model accuracy improved during training and validation across multiple epochs.

<img width="455" height="286" alt="image" src="https://github.com/user-attachments/assets/b20a3221-3247-4841-83d3-bf720432108c" />

**2- Training Loss Graph**

This graph shows how the training loss decreased during the learning process.

<img width="455" height="286" alt="image" src="https://github.com/user-attachments/assets/b67f412c-5724-4fbe-ba10-c49db19e163e" />

**3- Confusion Matrix**

The confusion matrix shows detailed classification performance of the CNN model for all four classes.

<img width="455" height="286" alt="image" src="https://github.com/user-attachments/assets/07e74bbb-9801-4636-aca6-8a02e512dec8" />

---

## **✨ Project Features**

Our system can:

- Train a CNN model on road images
- Detect multiple road conditions
- Predict custom test images
- Generate accuracy graphs
- Generate confusion matrix
- Save the best trained model automatically

---

## **📁 Project Structure**

```
PROJECT-2/
│
├── Dataset/
├── split_dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── train.py
├── test.py
├── predict.py
├── confusion_matrix.py
├── classification_report.py
├── split_dataset.py
│
├── best_road_model.pth
├── accuracy_graph.png
├── loss_graph.png
├── confusion_matrix.png
│
└── README.md
```

---

## **▶️ How to Run the Project**

**1. Install Required Libraries**
```c
pip install torch torchvision torchaudio
pip install matplotlib pillow scikit-learn seaborn
```

**2. Train the Model**
```c
python train.py
```

**3. Test the Model**
```c
python test.py
```

**4. Predict Custom Image**

Place a test image in the project folder and rename it:
```c
test.jpg
```

Then run:
```c
python predict.py
```

---

## **⚠️ Challenges We Faced**

During this project, we faced several challenges such as:

- collecting a balanced dataset
- handling different lighting conditions
- differentiating similar road textures
- training on limited hardware resources

Despite these challenges, we successfully developed a working AI-based road damage detection system.

## **🚀 Future Improvements**

In the future, this project can be improved by:

- using larger datasets
- using transfer learning models
- implementing real-time detection
- deploying as a web/mobile application
- integrating with smart city systems
- using drone or vehicle-mounted cameras

---

## **📝 Conclusion**

This project helped us understand:

- Deep Learning
- CNN architectures
- PyTorch implementation
- Image preprocessing
- Model evaluation techniques

The project demonstrates how AI can be used to solve real-world infrastructure problems and improve road safety.

---

## **👤 Contact**

If you have any questions or would like to discuss the project:
- Naimal Munir - nemalmunir@gmail.com
- Eman Fatima - emanfatima5978@gmail.com
- Fatima Muaz - 

