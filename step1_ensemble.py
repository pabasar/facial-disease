# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install necessary packages
!pip install torch torchvision

# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import StepLR

# Check if a GPU is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Set transforms
def set_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transforms, test_transforms

# Load data
def load_data(train_path, test_path, batch_size):
    train_transforms, test_transforms = set_transforms()

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Create model
def create_model(model_func, num_classes, unfreeze_layers=False):
    model = model_func(pretrained=True)
    if model_func == models.resnet50:
        if unfreeze_layers:
            for param in model.layer4.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_func == models.vgg16:
        for param in model.features.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_func == models.densenet121:
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

# Function to save the ensemble model
def save_ensemble_model(models, save_path):
    state_dicts = {f'model_{i}': model.state_dict() for i, model in enumerate(models)}
    torch.save(state_dicts, save_path)

# Train model
def train_model(model, train_loader, criterion, optimizer, scheduler, epochs, model_save_path):
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            torch.save(model.state_dict(), model_save_path)

        scheduler.step()

# Evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    overall_accuracy = correct / total
    print(f'Overall Test Accuracy: {overall_accuracy:.4f} ({correct}/{total})')
    return np.array(all_preds), np.array(all_labels)

# Plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes, rotation=45)
    plt.title('Confusion Matrix')
    plt.show()

# Load data
train_path = '/content/drive/MyDrive/facial_disease/dataset/train'
test_path = '/content/drive/MyDrive/facial_disease/dataset/test'
batch_size = 8
train_loader, test_loader = load_data(train_path, test_path, batch_size)

# Initialize models
num_classes = 4
resnet50 = create_model(models.resnet50, num_classes, unfreeze_layers=True)
vgg16 = create_model(models.vgg16, num_classes)
densenet121 = create_model(models.densenet121, num_classes)
ensemble_models = [resnet50, vgg16, densenet121]

# Define criterion and optimizer for each model
criterion = nn.CrossEntropyLoss()
optimizer_resnet50 = optim.Adam(resnet50.parameters(), lr=0.001)
optimizer_vgg16 = optim.Adam(vgg16.parameters(), lr=0.001)
optimizer_densenet121 = optim.Adam(densenet121.parameters(), lr=0.001)
scheduler_resnet50 = StepLR(optimizer_resnet50, step_size=20, gamma=0.1)
scheduler_vgg16 = StepLR(optimizer_vgg16, step_size=20, gamma=0.1)
scheduler_densenet121 = StepLR(optimizer_densenet121, step_size=20, gamma=0.1)

# Train models
train_model(resnet50, train_loader, criterion, optimizer_resnet50, scheduler_resnet50, epochs=60, model_save_path='/content/drive/MyDrive/facial_disease/code/ens_resnet50_model.pth')
train_model(vgg16, train_loader, criterion, optimizer_vgg16, scheduler_vgg16, epochs=60, model_save_path='/content/drive/MyDrive/facial_disease/code/ens_vgg16_model.pth')
train_model(densenet121, train_loader, criterion, optimizer_densenet121, scheduler_densenet121, epochs=60, model_save_path='/content/drive/MyDrive/facial_disease/code/ens_densenet121_model.pth')

# Evaluate each model
print("\nEvaluating ResNet-50")
predicted_labels_resnet50, true_labels_resnet50 = evaluate_model(resnet50, test_loader)
print("\nEvaluating VGG16")
predicted_labels_vgg16, true_labels_vgg16 = evaluate_model(vgg16, test_loader)
print("\nEvaluating DenseNet-121")
predicted_labels_densenet121, true_labels_densenet121 = evaluate_model(densenet121, test_loader)

# Ensemble prediction (simple averaging method)
def ensemble_predict(models, dataloader):
    model_resnet50, model_vgg16, model_densenet121 = models
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs_resnet50 = model_resnet50(inputs)
            outputs_vgg16 = model_vgg16(inputs)
            outputs_densenet121 = model_densenet121(inputs)
            outputs = (outputs_resnet50 + outputs_vgg16 + outputs_densenet121) / 3
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    overall_accuracy = correct / total
    print(f'Ensemble Model Test Accuracy: {overall_accuracy:.4f} ({correct}/{total})')
    return np.array(all_preds), np.array(all_labels)

# Evaluate ensemble model
predicted_labels_ensemble, true_labels_ensemble = ensemble_predict((resnet50, vgg16, densenet121), test_loader)

# After training, save the ensemble model
save_ensemble_model(ensemble_models, '/content/drive/MyDrive/facial_disease/code/ensemble_model.pth')

# Plot confusion matrix and classification report for the ensemble
plot_confusion_matrix(true_labels_ensemble, predicted_labels_ensemble, train_loader.dataset.classes)
print('\nEnsemble Model Classification Report\n')
print(classification_report(true_labels_ensemble, predicted_labels_ensemble, target_names=train_loader.dataset.classes))

