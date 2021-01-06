"""
This script contains the transformation needed for the Mask-RCNN by adapting the FastRCNNPredictor &
an autoencoder architecture that returns the reconstructed image and the latent vector for analysis.
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torch.nn as nn
import torch.nn.functional as F


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


class ConvAutoencoder(nn.Module):

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 5, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 5, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(3, 3)
        # Final linear layer of encoder to reduce data to 100 latent values
        self.linear1 = nn.Linear(44556, 100)
        ## Decoder layers ##
        ## Linear layer to pass data into image format
        self.linear2 = nn.Linear(100, 230400)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation - torch.Size([1, 4, 180, 320])
        x = x.view(-1)
        x_out = F.relu(self.linear1(x))

        ## decode ##
        x = F.relu(self.linear2(x_out))
        # Convert vector into 2D array (Batch,  channels,  Height, Width)
        x = x.view(1, 4, 180, 320)
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))

        return x, x_out
