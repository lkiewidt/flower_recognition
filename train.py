# imports
import argparse
import utilities as utl
from torchvision import models
import torch
from torch import nn, optim
import torch.nn.functional as F

# define command line arguments
parser = argparse.ArgumentParser(description='Image classifier options')

# define arguments
# a. data and save directories
parser.add_argument('data_directory', action='store',
                    help='Path to directory with labelled image dataset')

parser.add_argument('--save_dir', action='store',
                    dest='save_dir', default='chechpoints',
                    help='Directory to save checkpoints')

# b. hyperparameters
parser.add_argument('--arch', action='store',
                    dest='architechture', default='vgg16',
                    help='architecture of pre-trained model')

parser.add_argument('--learning_rate', action='store', type=float,
                    dest='learning_rate', default=0.001,
                    help='optimizer learning rate')

parser.add_argument('--hidden_units', action='store', type=int,
                    dest='hidden_units', default=4096,
                    help='number of units in hidden layer')

parser.add_argument('--dropout', action='store', type=float,
                    dest='dropout', default=0.25,
                    help='dropout probability')

parser.add_argument('--epochs', action='store', type=int,
                    dest='n_epochs', default=3,
                    help='number of epochs')

parser.add_argument('--print', action='store', type=int,
                    dest='print_steps', default=10,
                    help='print training information after this number of steps')

parser.add_argument('--gpu', action='store_true',
                    dest='gpu', default=False,
                    help='gpu switch')

# parse arguments
arguments = parser.parse_args()

# load datasets
print("+ Pre-processing images...", end=' ')
train_loader, valid_loader, test_loader, class_to_idx = utl.load_data(arguments.data_directory)
print("COMPLETED")

# build network
print("+ Loading and building neural network...", end=' ')
model = getattr(models, arguments.architechture)(pretrained=True)

# freeze feature parameters of pretrained model
for param in model.parameters():
    param.requires_grad = False

# build classifier
classifier = nn.Sequential(nn.Linear(25088, arguments.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(p=arguments.dropout),
                           nn.Linear(arguments.hidden_units, 102),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier
model.class_to_idx = class_to_idx
print("COMPLETED")

# train network
# set device
if arguments.gpu and torch.cuda.is_available():
    device = 'cuda'
    gpu_msg = "using GPU"
elif arguments.gpu:
    device = 'cpu'
    gpu_msg = "GPU not found! Using CPU instead"
else:
    device = 'cpu'
    gpu_msg = "using CPU"
    
model.to(device)

# create loss criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)

# start training network
print("")
print("+ Training network: " + gpu_msg)
print("+ Epochs = {}, ".format(arguments.n_epochs),
      "arch = {}, ".format(arguments.architechture),
      "hidden units = {}, ".format(arguments.hidden_units),
      "dropout = {:.3E}, ".format(arguments.dropout),
      "learning rate = {:.3E}, ".format(arguments.learning_rate))

model, optimizer = utl.train(model, train_loader, valid_loader, criterion, optimizer,
                             arguments.n_epochs, device, print_steps=arguments.print_steps)

# test network
test_loss, test_accuracy = utl.validation(model, test_loader, criterion, device)
print("")
print("+ Training COMPLETED: final test accuracy = {:.3f}".format(test_accuracy))

# save model
# create checkpoint
checkpoint = {'model_name': arguments.architechture,
              'model_state': model.state_dict(),
              'model_classifier': model.classifier,
              'n_epochs': arguments.n_epochs,
              'optim_state': optimizer.state_dict(),
              'class_to_idx': class_to_idx}

# save checkpoint
filename = "%s_%i_%i-epochs.pth" % (arguments.architechture, arguments.hidden_units, arguments.n_epochs)
print("")
print("+ Saving network to %s/%s" % (arguments.save_dir, filename))
torch.save(checkpoint, arguments.save_dir + '/' + filename)