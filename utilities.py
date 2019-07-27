# imports
import argparse
import numpy as np
import torch
from torchvision import transforms, datasets, models
from PIL import Image



def load_data(data_dir):
    
    # set path train, validation, and test images
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # image properties and requirements
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    img_size = 224

    # training image transformations
    train_transf = transforms.Compose([transforms.RandomRotation(30),    # rotate randomly up to 30°
                                       transforms.RandomCrop(img_size),  # crop to img_size x img_size at random position 
                                       transforms.RandomVerticalFlip(),  # flip vertically with 50% chance
                                       transforms.ToTensor(),
                                       transforms.Normalize(img_mean, img_std)])

    # validation and test image transformations
    valtest_transf = transforms.Compose([transforms.CenterCrop(img_size),  # crop to img_size x img_size at random position 
                                         transforms.ToTensor(),
                                         transforms.Normalize(img_mean, img_std)])

    # load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transf)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valtest_transf)
    test_datasets = datasets.ImageFolder(test_dir, transform=valtest_transf)

    # using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True) 
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64) 
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64)

    return (train_loader, valid_loader, test_loader, train_datasets.class_to_idx)


# define validation function
def validation(model, dataloader, criterion, device):
    running_valid_loss = 0.0
    running_accuracy = 0.0

    # set to evaluation mode
    model.to(device)
    model.eval()
        
    # loop over validation batches
    with torch.no_grad():
        for images, labels in dataloader:
        
            images, labels = images.to(device), labels.to(device)
        
            # predict (feedforward)
            log_ps = model.forward(images)
        
            # calculate loss
            batch_loss = criterion(log_ps, labels)
            running_valid_loss += batch_loss.item()

            # calculate accuracy
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    valid_loss = running_valid_loss/len(dataloader)
    accuracy = running_accuracy/len(dataloader)
    
    return (valid_loss, accuracy)


def train(model, train_loader, valid_loader, criterion, optimizer, n_epochs, device, print_steps=10):
    steps = 0
    
    for e in range(n_epochs):
        running_train_loss = 0.0
        model.train()

        for images, labels in train_loader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            # set gradients to 0
            optimizer.zero_grad()

            # feedforward
            log_ps = model.forward(images)

            # calculate loss and cumulate losses for all training datasets
            loss = criterion(log_ps, labels)
            running_train_loss += loss.item()

            # back-propagation
            loss.backward()

            # update weights
            optimizer.step()

            if steps % print_steps == 0:
                valid_loss, accuracy = validation(model, valid_loader, criterion, device)

                print("+-- Epoch {}/{}, ".format(e+1, n_epochs),
                      "training loss = {:.4E}, ".format(running_train_loss/len(train_loader)),
                      "validation loss = {:.4E}, ".format(valid_loss),
                      "accuracy = {:.3f}".format(accuracy))

                running_train_loss = 0.0
                model.train()

    return (model, optimizer)


def load_image(image_path):
    # load image
    img = Image.open(image_path)
    
    # define transformations
    # transformations on PIL image can conveniently be done with transforms
    img_transforms = transforms.Compose([transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    # apply transformations
    img_transf = img_transforms(img)
    
    return img_transf


def make_prediction(image_path, model, topk, device):
    # load and pre-process image
    image = load_image(image_path)
    image = image.to(device)
    
    # predict
    model.eval()
    model.to(device)
    with torch.no_grad():
        log_ps = model.forward(image.unsqueeze(0))
        ps = torch.exp(log_ps)
        
        top_p, top_label = ps.topk(topk, dim=1)
        
    # convert to numpy
    top_p = top_p.cpu().numpy().squeeze()
    top_label = top_label.cpu().numpy().squeeze()
    
    return (top_p, top_label)