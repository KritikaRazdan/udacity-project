import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from train import load_model  
import json
import argparse


def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name.')
    
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--labels', type=str, default='')
    parser.add_argument('--gpu', action='store_true')

    return parser.parse_args()


def predict(image_path, checkpoint_path, topk=5, label_path='', gpu=False):
   
    
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
   
    model = load_model(checkpoint_path)   
   
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    
    model.eval()
    
    
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
   
    pil_image = Image.open(image_path)
    pil_image = img_loader(pil_image).float()
    
    
    image = torch.FloatTensor([np.array(pil_image)]).to(device)

    
    with torch.no_grad():
        output = model(image)
        probabilities, classes = output.topk(topk)
        probabilities = torch.nn.functional.softmax(probabilities, dim=1)
    probabilities = probabilities.cpu().numpy()[0]
    classes = classes.cpu().numpy()[0]
    print(classes)
    print(probabilities)
   
    if label_path:
        with open(label_path, 'r') as f:
            cat_to_name = json.load(f)
        
        class_names = []
        for abc in classes:
            for key, value in checkpoint['class_to_idx'].items():
                if value == abc:
                    class_names.append(key)
    else:
        class_names = classes  

    return probabilities, class_names

def main():
    args = get_input_args()
   
    model = load_model(args.checkpoint)
    
    probs, classes = predict(args.image, args.checkpoint, args.topk, args.labels, args.gpu)
    
    print(f"Top {args.topk} predictions and probabilities for {args.image}:")
    for cls, prob in zip(classes, probs):
        print(f"{cls}: {prob:.4f}")

if __name__ == '__main__':
    main()

