# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install necessary packages
!pip install grad-cam

# Import required libraries
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Check if a GPU is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Function to load the ensemble model
def load_ensemble_model(model_path, device):
    ensemble_state_dict = torch.load(model_path, map_location=device)
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(ensemble_state_dict['model_0'])  # Load ResNet50 part
    model = model.to(device)
    model.eval()
    return model

# Function to find the last convolutional layer in ResNet50 and return the layer
def find_resnet50_last_conv_layer(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_layer = module
    return last_conv_layer

# Function to preprocess the image
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Function to apply Grad-CAM
def apply_gradcam(model, img_tensor, target_layer):
    grad_cam = GradCAM(model=model, target_layers=[target_layer])
    model_output = model(img_tensor)
    target_index = model_output.argmax(dim=1).item()

    # Wrap the target_index in a list of targets
    targets = [ClassifierOutputTarget(target_index)]

    # Generate the Grad-CAM heatmap
    cam = grad_cam(img_tensor, targets=targets)[0, :]
    return cam, target_index

# Function to superimpose Grad-CAM heatmap on the original image
def superimpose(img_path, cam):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # Normalize the CAM data to 0-255 and convert to unsigned 8-bit integer
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)

    # Apply color map to the heatmap
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # Convert to float for further calculations
    heatmap = np.float32(heatmap)
    img = np.float32(img)

    # Superimpose the heatmap onto the original image with a specific opacity
    superimposed_img = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)

    # Convert the superimposed image to the format suitable for display (uint8)
    superimposed_img = np.uint8(superimposed_img)

    # Convert BGR to RGB for display with matplotlib
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img

# Function to predict and visualize Grad-CAM
def predict_and_visualize_gradcam(model, img_path, device, class_names):
    img_tensor = preprocess_image(img_path).to(device)
    # Get model output and calculate softmax probabilities
    model_output = model(img_tensor)
    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(model_output)
    conf_score, target_index = torch.max(probs, dim=1)
    predicted_class = class_names[target_index.item()]
    conf_percentage = conf_score.item() * 100  # Convert to percentage
    last_conv_layer = find_resnet50_last_conv_layer(model)
    cam, _ = apply_gradcam(model, img_tensor, last_conv_layer)
    superimposed_img = superimpose(img_path, cam)
    plt.imshow(superimposed_img)
    plt.title(f'{predicted_class}, {conf_percentage:.2f}%')
    plt.axis('off')
    plt.show()

# Load the model and other variables
model_path = '/content/drive/MyDrive/facial_disease/code/ensemble_model.pth'
model = load_ensemble_model(model_path, device)
class_names = ["Bell's Palsy", "Healthy", "Moebius Syndrome", "Parry-Romberg Syndrome"]

# Call the visualization function
predict_and_visualize_gradcam(model, '/content/drive/MyDrive/facial_disease/code/visualization/test1.jpg', device, class_names)
predict_and_visualize_gradcam(model, '/content/drive/MyDrive/facial_disease/code/visualization/test2.jpg', device, class_names)
predict_and_visualize_gradcam(model, '/content/drive/MyDrive/facial_disease/code/visualization/test3.jpg', device, class_names)
predict_and_visualize_gradcam(model, '/content/drive/MyDrive/facial_disease/code/visualization/test4.jpg', device, class_names)
