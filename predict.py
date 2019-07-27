# imports
import argparse
import utilities as utl
from torchvision import models
import torch
import json

# define command line arguments
parser = argparse.ArgumentParser(description='Image classifier options')

# define arguments
parser.add_argument('image', action='store',
                    help='Path to input image')

parser.add_argument('checkpoint', action='store',
                    help='Path to network checkpoint to be used for prediction')

parser.add_argument('--top_k', action='store', type=int,
                    dest='topk', default=5,
                    help='number of classes to be shown in the prediction')

parser.add_argument('--category_names', action='store',
                    dest='categories', default=None,
                    help='definition of alternative class names')

parser.add_argument('--gpu', action='store_true',
                    dest='gpu', default=False,
                    help='gpu switch')

# parse arguments
arguments = parser.parse_args()

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
    
# load checkpoint and build model
checkpoint = torch.load(arguments.checkpoint)
model = getattr(models, checkpoint['model_name'])(pretrained=True)
model.classifier = checkpoint['model_classifier']
model.load_state_dict(checkpoint['model_state'])
class_to_label = checkpoint['class_to_idx']

# reverse class_to_label dictionary
label_to_class = {val: key for key, val in class_to_label.items()}

# predict
top_p, top_label = utl.make_prediction(arguments.image, model, arguments.topk, device)

# get top classes from top labels
top_class = [label_to_class[key] for key in top_label]

# print results
# load alternative category names
if arguments.categories is not None:
    with open(arguments.categories, 'r') as f:
        class_to_name = json.load(f)
    
    # get class names for top classes
    top_class_name = [class_to_name[key] for key in top_class]
    
print("+ Predictions for %s" % arguments.image)
if arguments.categories is None:
    print("+ RANK\tPROB.\tCLASS")
    for i in range(arguments.topk):
        print("+ %i.\t%.3f\t%s" % (i+1, top_p[i], top_class[i]))
else:
    print("+ RANK\tPROB.\tCLASS\tCLASS NAME")
    for i in range(arguments.topk):
        print("+ %i.\t%.3f\t%s\t%s" % (i+1, top_p[i], top_class[i], top_class_name[i]))
        