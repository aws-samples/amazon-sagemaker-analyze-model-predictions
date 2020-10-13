'''SageMaker PyTorch inference container overrides to serve ResNet18 model with debug hook'''

# Python Built-Ins:
import argparse
from io import BytesIO
import logging
import os
from typing import Any, Union

# External Dependencies:
import numpy as np
from PIL import Image
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.core.modes import ModeKeys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

# Local Dependencies:
from custom_hook import CustomHook


logger = logging.getLogger()


class ModelWithDebugHook:
    def __init__(self, model: Any, hook: Union[smd.Hook, None]):
        '''Simple container to associate a 'model' with a SageMaker debug hook'''
        self.model = model
        self.hook = hook


def model_fn(model_dir: str) -> ModelWithDebugHook:
    #create model    
    model = models.resnet18()

    #traffic sign dataset has 43 classes   
    nfeatures = model.fc.in_features
    model.fc = nn.Linear(nfeatures, 43)

    #load model
    weights = torch.load(f'{model_dir}/model/model.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(weights)

    model.eval()
    model.cpu()

    #hook configuration
    tensors_output_s3uri = os.environ.get('tensors_output')
    if tensors_output_s3uri is None:
        logger.warning(
            'WARN: Skipping hook configuration as no tensors_output env var provided. '
            'Tensors will not be exported'
        )
        hook = None
    else:
        save_config = smd.SaveConfig(mode_save_configs={
            smd.modes.PREDICT: smd.SaveConfigMode(save_interval=1),
        })

        hook = CustomHook(
            tensors_output_s3uri,
            save_config=save_config,
            include_regex='.*bn|.*bias|.*downsample|.*ResNet_input|.*image|.*fc_output',
        )

        #register hook
        hook.register_module(model) 

        #set mode
        hook.set_mode(modes.PREDICT)

    return ModelWithDebugHook(model, hook)


def transform_fn(model_with_hook, data, content_type, output_content_type):
    model = model_with_hook.model
    hook = model_with_hook.hook

    val_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = np.load(BytesIO(data))
    image = Image.fromarray(image)
    image = val_transform(image)

    image = image.unsqueeze(0)
    image = image.to('cpu').requires_grad_()
    if hook is not None:
        hook.image_gradients(image)

    #forward pass
    prediction = model(image)

    #get prediction
    predicted_class = prediction.data.max(1, keepdim=True)[1]
    output = prediction[0, predicted_class[0]]
    model.zero_grad()

    #compute gradients with respect to outputs 
    output.backward()

    response_body = np.array(predicted_class.cpu()).tolist()
    return response_body, output_content_type
