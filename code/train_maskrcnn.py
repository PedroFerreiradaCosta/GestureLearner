"""
This script trains a Mask R-CNN on the EgoHands dataset (http://vision.soic.indiana.edu/projects/egohands/)
to learn to segment multiple hands from images. It also learns to predict boudning boxes,  as it is based
on top of the Faster R-CNN.
The code is adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.

"""


# %%shell
#
# # Download TorchVision repo to use some files from
# # references/detection
# git clone https://github.com/pytorch/vision.git
# cd vision
# git checkout v0.3.0
#
# cp references/detection/utils.py ../
# cp references/detection/transforms.py ../
# cp references/detection/coco_eval.py ../
# cp references/detection/engine.py ../
# cp references/detection/coco_utils.py ../


# Data augmentation vertical flip
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torch
import  os
from data_reader import EgoHandsDataset
from network import get_instance_segmentation_model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# use our dataset and defined transformations
dataset = EgoHandsDataset('../data/', get_transform(train=True))
dataset_test = EgoHandsDataset('../data/', get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
torch.backends.cudnn.deterministics = True
torch.backends.cudnn.benchmark = False
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-100])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

# Define GPUs
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using {torch.cuda.device_count()} GPUs')

num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model = torch.nn.DataParallel(model).to(device)

# construct an optimizer
###  Check optimizer with Walter
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 1 epoch
num_epochs = 1

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)


print('Finished Training')
os.makedirs('../models', exist_ok=True)
torch.save(model, '../models/model_segmentation.torch')
