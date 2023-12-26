# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install necessary packages
!pip install captum

# Import required libraries
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import matplotlib.pyplot as plt

# Check if a GPU is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Function to create a model
def create_model(model_func, num_classes):
    model = model_func(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Adjusting the classifier for different model architectures
    if model_func == models.vgg16:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_func == models.densenet121:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        # Default case, e.g., for ResNet
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model.to(device)

# Function to load a model
def load_model(model_func, model_path, num_classes, device, model_key):
    model = create_model(model_func, num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict[model_key])
    model.eval()
    return model

# Load each model from the ensemble
ensemble_path = '/content/drive/MyDrive/facial_disease/code/ensemble_model.pth'
num_classes = 4  
resnet50 = load_model(models.resnet50, ensemble_path, num_classes, device, 'model_0')
vgg16 = load_model(models.vgg16, ensemble_path, num_classes, device, 'model_1')
densenet121 = load_model(models.densenet121, ensemble_path, num_classes, device, 'model_2')
ensemble_models = [resnet50, vgg16, densenet121]

# Function to preprocess the image
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Function to visualize the attributions along with the original image
def visualize_attr(input_image, attr, main_title, subtitles):
    # Convert the input image tensor to numpy and remove the batch dimension
    input_image_numpy = input_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

    # Convert the attribution tensor to numpy and remove the batch dimension
    attr_numpy = attr.squeeze().cpu().detach().numpy().transpose(1, 2, 0)

    # Use Captum's visualize_image_attr_multiple for multiple visualizations
    fig, _ = viz.visualize_image_attr_multiple(
        attr_numpy,
        input_image_numpy,
        methods=["original_image", "heat_map", "blended_heat_map"],
        signs=["all", "absolute_value", "all"],
        show_colorbar=True,
        titles=subtitles
    )

    # Set the main title for the visualization
    fig.suptitle(main_title, fontsize=16)

# Function for prediction and Captum visualization
def predict_and_visualize(image_path, models, class_names, target_class_idx=None, n_steps=50):
    input_image = preprocess_image(image_path).to(device)
    input_image.requires_grad = True

    # Ensemble Prediction
    outputs = []
    for model in models:
        output = model(input_image)
        output = nn.functional.softmax(output, dim=1)
        outputs.append(output)
    avg_output = torch.mean(torch.stack(outputs), dim=0)
    prediction_score, pred_class_idx = torch.max(avg_output, 1)

    if target_class_idx is None:
        target_class_idx = pred_class_idx.item()

    predicted_class = class_names[pred_class_idx.item()]
    prediction_percentage = prediction_score.item() * 100

    print(f'\033[1mCondition: {predicted_class}, Score: {prediction_percentage:.2f}%\033[0m')

    # Integrated Gradients
    integrated_gradients = IntegratedGradients(models[0])  # Using the first model in the ensemble for visualization
    attributions_ig = integrated_gradients.attribute(input_image, target=target_class_idx, n_steps=n_steps)

    # Visualization
    main_title = f'{predicted_class} ({prediction_percentage:.2f}%)'
    subtitles = ["Original Image", "Attribution Magnitude", "Overlayed Gradients"]
    visualize_attr(input_image, attributions_ig, main_title, subtitles)

class_names = ["Bell's Palsy", "Healthy", "Moebius Syndrome", "Parry-Romberg Syndrome"]

# Testing
predict_and_visualize('/content/drive/MyDrive/facial_disease/code/visualization/test1.jpg', ensemble_models, class_names)
predict_and_visualize('/content/drive/MyDrive/facial_disease/code/visualization/test2.jpg', ensemble_models, class_names)
predict_and_visualize('/content/drive/MyDrive/facial_disease/code/visualization/test3.jpg', ensemble_models, class_names)
predict_and_visualize('/content/drive/MyDrive/facial_disease/code/visualization/test4.jpg', ensemble_models, class_names)

