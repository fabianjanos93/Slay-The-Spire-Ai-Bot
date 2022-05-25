# Import Libraries
from datetime import datetime

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler

# example from : https://towardsdatascience.com/a-beginners-tutorial-on-building-an-ai-image-classifier-using-pytorch-6f85cb69cba7

transformations = transforms.Compose([
    transforms.Resize(120),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder("root/train", transform=transformations)
val_set = datasets.ImageFolder("root/test", transform=transformations)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=True)

# Get pretrained model using torchvision.models as models library
model = models.densenet161(pretrained=True)

# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False

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

# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Move model to the device specified above
model.to(device)

# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
# Set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.classifier.parameters())


epochs = 1000
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0

    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item() * inputs.size(0)

        # Print the progress of our training
        counter += 1
        print(datetime.now().strftime("%H:%M:%S"), counter, "/", len(train_loader))

    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = criterion(output, labels)
            # Add loss to the validation set's running loss
            val_loss += valloss.item() * inputs.size(0)

            # Since our model outputs a LogSoftmax, find the real
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Print the progress of our evaluation
            counter += 1
            print(datetime.now().strftime("%H:%M:%S"), counter, "/", len(val_loader))

    # Get the average loss for the entire epoch
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = val_loss / len(val_loader.dataset)
    # Print out the information
    torch.save(model.state_dict(), "playerHandModel-"+str(epoch)+".model")
    print('Accuracy: ', accuracy / len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    print("Current Time =", datetime.now().strftime("%H:%M:%S"))

# ---------------TEST -----------------
model.eval()


# Process our image
def process_image(image_path):
    # Load Image
    img = Image.open(image_path)

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


# Process Image
image = process_image("root/image1234.jpg")
# Give image to model to predict output
top_prob, top_class = predict(image, model)
# Show the image
show_image(image)
# Print the results
print("The model is ", top_prob * 100, "% certain that the image has a predicted class of ", top_class)

torch.save(model.state_dict(), "playerHandModel.model")


# 11. epoch ~58% accuracy
