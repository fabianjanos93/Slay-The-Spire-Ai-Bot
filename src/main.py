import time
from random import randint

import PIL
import numpy as np
import torch
from PIL import ImageGrab
import pytesseract as pytesseract
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2

from src.card import Card


def process_image(img):
    # Get the dimensions of the image
    width, height = img.size

    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    img = img.resize((120, int(120 * (height / width))) if width < height else (int(120 * (width / height)), 120))

    # Get the dimensions of the new image size
    width, height = img.size

    # Set the coordinates to do a center crop of 224 x 224
    # left = (width - 224) / 2
    # top = (height - 224) / 2
    # right = (width + 224) / 2
    # bottom = (height + 224) / 2
    # img = img.crop((left, top, right, bottom))

    # Turn image into numpy array
    img = np.array(img)

    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))

    # Make all values between 0 and 1
    img = img / 255

    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485) / 0.229
    img[1] = (img[1] - 0.456) / 0.224
    img[2] = (img[2] - 0.406) / 0.225

    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis, :]

    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image


# Using our model to predict the label
def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)

    # Reverse the log function in our output
    output = torch.exp(output)

    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()


# Show Image
def show_image(image):
    # Convert image to numpy
    image = image.numpy()

    # Un-normalize the image
    image[0] = image[0] * 0.226 + 0.445

    # Print the image
    fig = plt.figure(figsize=(25, 4))
    plt.imshow(np.transpose(image[0], (1, 2, 0)))


model = models.densenet161(pretrained=True)

# Create new classifier for model using torch.nn as nn library
classifier_input = model.classifier.in_features
num_labels = 11  # PUT IN THE NUMBER OF LABELS IN YOUR DATA
classifier = nn.Sequential(nn.Linear(classifier_input, 512),
                           nn.ReLU(),
                           nn.Linear(512, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))
# Replace default classifier with new classifier
model.classifier = classifier

model.load_state_dict(torch.load("C:/Programming/SlayTheSpireAIBot/src/playerHandModel-39.model"))
model.eval()


def get_number_of_cards():
    my_screenshot = PIL.ImageGrab.grab()
    my_screenshot = PIL.ImageGrab.grab(bbox=(0, 800, my_screenshot.width, my_screenshot.height))
    prediction = predict(process_image(my_screenshot), model)
    probability, folder_number = prediction
    return folder_number if folder_number < 2 else folder_number-1


