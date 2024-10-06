import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json


def get_input_args():
    parser = argparse.ArgumentParser(description='Train a flower classifier ')

    
    parser.add_argument('--data_dir', type=str, default='flowers')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth')
    parser.add_argument('--arch', type=str, default='vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    
    return parser.parse_args()

def data_transforms():
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, valid_transform

def load_data(data_dir, train_transform, valid_transform):
    train_directory = f'{data_dir}/train'
    valid_directory = f'{data_dir}/valid'
    test_directory = f'{data_dir}/test'

    train_dataset = datasets.ImageFolder(train_directory, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_directory, transform=valid_transform)
    test_dataset = datasets.ImageFolder(test_directory, transform=valid_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    return train_dataloader, valid_dataloader, test_dataloader

def initialize_model(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        
        model = models.vgg16(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    return model

def validate(model, dataloader, criterion, device):
    model.eval()  
    loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss += criterion(output, labels).item()
            probabilities = torch.exp(output)
            equality = (labels.data == probabilities.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss / len(dataloader), accuracy / len(dataloader)

def train_model(model, trainloader, validloader, criterion, optimizer, epochs, device, print_every=40):
    steps = 0
    for epoch in range(epochs):
        model.train()  
        running_loss = 0
        
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss, valid_accuracy = validate(model, validloader, criterion, device)
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train Loss: {running_loss/print_every:.3f}.. "
                      f"Valid Loss: {valid_loss:.3f}.. "
                      f"Valid Accuracy: {valid_accuracy:.3f}")
                
                running_loss = 0

def save_checkpoint(model, train_dataset, save_dir):
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'arch': 'vgg16'
    }
    torch.save(checkpoint, save_dir)
def load_model(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def main():
    args = get_input_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    
    train_transform, valid_transform = data_transforms()
    trainloader, validloader, testloader = load_data(args.data_dir, train_transform, valid_transform)

    model = initialize_model(args.arch, args.hidden_units)
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    
    train_model(model, trainloader, validloader, criterion, optimizer, args.epochs, device)

    save_checkpoint(model, trainloader.dataset, args.save_dir)

    load_model('checkpoint.pth')

if __name__ == '__main__':
    main()
