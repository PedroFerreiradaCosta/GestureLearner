import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from network import ConvAutoencoder

# Define GPUs
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using {torch.cuda.device_count()} GPUs')

# Model Autoencoder that we are training
model = ConvAutoencoder()
model = torch.nn.DataParallel(model).to(device)
# specify loss function
criterion = nn.BCELoss()

# Model to create a mask from the original image - to be fed  to the autoencoder
model_seg = torch.load('models/model_segmentation.torch')
model_seg = torch.nn.DataParallel(model).to(device)

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 10

for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0

    ###################
    # train the model #
    ###################
    img_path_list = list(sorted(os.listdir(os.path.join('data', "images"))))
    cml_loss = 0
    for i, img_path in enumerate(img_path_list):

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        img_path = os.path.join("data", "images", img_path)
        img = Image.open(img_path).convert("RGB")
        img_torch = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255

        model_seg.eval()
        with torch.no_grad():
            segmentation = model_seg([img_torch.to(device)])

        try:
            input = segmentation[0]['masks'][0, 0].unsqueeze(0)
        # In case no mask is generated, skip
        except IndexError:
            print(f"{i}: {IndexError}")
            continue
        input = input.unsqueeze(0)
        model.to(device)
        outputs = model(input)
        # calculate the loss
        loss = criterion(outputs, input)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * input.size(0)
        cml_loss += loss
        if i % 100 == 0:
            print(f"{i}: Loss: {cml_loss / 100}")
            cml_loss = 0
    # print avg training statistics
    train_loss = train_loss / len(img_path_list)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

print('Finished Training')
os.makedirs('../models', exist_ok=True)
torch.save(model, '../models/model_ae.torch')