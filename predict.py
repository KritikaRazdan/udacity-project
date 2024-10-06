import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from train import load_model  # Assumed to be a function that loads the model from the checkpoint
import json
import argparse

# Define command line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name.')
    
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint to use when predicting')
    parser.add_argument('--topk', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--labels', type=str, default='', help='Path to a JSON file mapping labels to flower names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    return parser.parse_args()

# Prediction function
def predict(image_path, checkpoint_path, topk=5, label_path='', gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
   
    model = load_model(checkpoint_path)   
    # Use GPU if selected and available
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Switch model to evaluation mode
    model.eval()
    
    # Define image transformation pipeline
    img_loader = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Process the image
    pil_image = Image.open(image_path)
    pil_image = img_loader(pil_image).float()
    
    # Convert to tensor and add batch dimension
    image = torch.FloatTensor([np.array(pil_image)]).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(image)
        probabilities, classes = output.topk(topk)
        probabilities = torch.nn.functional.softmax(probabilities, dim=1)
    probabilities = probabilities.cpu().numpy()[0]
    classes = classes.cpu().numpy()[0]
    print(classes)
    print(probabilities)
    # Map the class indices to actual class names using the label mapping file if provided
    if label_path:
        with open(label_path, 'r') as f:
            cat_to_name = json.load(f)
        
        class_names = []
        for abc in classes:
            for key, value in checkpoint['class_to_idx'].items():
                if value == abc:
                    class_names.append(key)
    else:
        class_names = classes  # Use raw class indices if no label mapping file

    return probabilities, class_names

# Main function to handle predictions from the command line
def main():
    # Get command line arguments
    args = get_input_args()
    
    # Load the model using the checkpoint
    model = load_model(args.checkpoint)
    
    # Perform the prediction
    probs, classes = predict(args.image, args.checkpoint, args.topk, args.labels, args.gpu)
    
    # Print the results
    print(f"Top {args.topk} predictions and probabilities for {args.image}:")
    for cls, prob in zip(classes, probs):
        print(f"{cls}: {prob:.4f}")

# Run the main function when the script is executed directly
if __name__ == '__main__':
    main()

