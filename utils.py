# Python Built-Ins:
import os

# External Dependencies:
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# Configuration:
image_norm_mean = [0.485, 0.456, 0.406]
image_norm_stddev = [0.229, 0.224, 0.225]


def get_dataloader():
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(image_norm_mean, image_norm_stddev),
    ])
    dataset = datasets.ImageFolder('GTSRB/Final_Test/', val_transform)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    return val_dataloader


def tensor_to_imgarray(image, floating_point=False):
    """Convert a normalized tensor or matrix as used by the model into a standard image array

    Parameters
    ----------
    image : Union[numpy.ndarray, torch.Tensor]
        A mean/std-normalized image tensor or matrix in inference format for the model
    floating_point : bool (Optional)
        Set True to skip conversion to 0-255 uint8 and return a 0-1.0 float ndarray instead
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
        if len(image.shape) > 3:
            # Leading batch dimension - take first el only
            image = image[tuple(0 if dim == 0 else slice(None) for dim in range(len(image.shape)))]

    image_shape = image.shape
    channeldim = image_shape.index(3)
    result = image

    # Move channel to correct (trailing) dim if not already:
    if channeldim < (len(image_shape) - 1):
        result = np.moveaxis(result, channeldim, -1)
        image_shape = result.shape
        channeldim = len(image_shape) - 1

    # Pad mean and stddev constants to image dimensions
    # TODO: Simplify this when we're consistent in what the image dimensions are!
    stddev = np.expand_dims(
        image_norm_stddev,
        list(range(channeldim)) + list(range(channeldim + 1, len(image_shape)))
    )
    mean = np.expand_dims(
        image_norm_mean,
        list(range(channeldim)) + list(range(channeldim + 1, len(image_shape)))
    )

    result = (result * stddev) + mean
    if floating_point:
        return np.clip(result, 0., 1.)
    else:
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def load_model():
    #check if GPU is available and set context
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #load model
    model = models.resnet18()

    #traffic sign dataset has 43 classes
    nfeatures = model.fc.in_features
    model.fc = nn.Linear(nfeatures, 43)

    weights = torch.load('model/model.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(weights)

    for param in model.parameters():
        param.requires_grad = False

    model.to(device).eval()
    return model


def show_images_diff(image, adv_image, adv_label=None, class_names=None, cmap=None):
    adv_image = tensor_to_imgarray(adv_image, floating_point=True)
    image = tensor_to_imgarray(image, floating_point=True)

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))

    ax0.imshow(image)
    ax0.set_title('Original')
    ax0.set_axis_off()

    ax1.imshow(adv_image)
    if adv_label is None:
        ax1.set_title('Adversarial image')
    else:
        ax1.set_title(f'Model prediction: {class_names[adv_label] if class_names else adv_label}')
    ax1.set_axis_off()

    difference = adv_image - image

    # If colormapping, convert RGB to single lightness channel:
    if cmap is not None and 3 in difference.shape:
        channeldim = difference.shape.index(3)
        rgbindices = [
            tuple(rgb if dim == channeldim else slice(None) for dim in range(len(difference.shape)))
            for rgb in range(3)
        ]
        # RGB->lightness function per PIL docs, but no need to import the lib just for this:
        # https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        difference = (
            difference[rgbindices[0]] * 0.299
            + difference[rgbindices[1]] * 0.587
            + difference[rgbindices[2]] * 0.114
        )

    # Scale to a symmetric range around max absolute difference (which we print out), and map that to 0-1
    # for imshow. (When colormapping we could just use vmin/vmax, but this way we keep same path for both).
    maxdiff = abs(difference).max()
    difference = difference / (maxdiff * 2.0) + 0.5
    ax2.imshow(difference, cmap=cmap, vmin=0., vmax=1.)
    ax2.set_title(f'Diff ({-maxdiff:.4f} to {maxdiff:.4f})')
    ax2.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_saliency_map(
    saliency_map,
    image,
    predicted_class=None,
    class_names=None,
    confidence=None,
    cmap=plt.cm.plasma,
    alpha=0.5,
):
    """Plot an image classification result with saliency map

    Parameters
    ----------
    saliency_map :
        A *normalized* (range 0-1.0) importance/saliency map matching image height and width, but with no
        channel dimension.
    image :
        An image with leading channel dimension, normalized values (mean + std).
        TODO: Parameterize the normalization rather than hard-coding?
    predicted_class : Any (Optional)
        If supplied, the saliency overlay plot will be titled to indicate which class was detected.
    class_names : Mapping[Any, Any] (Optional)
        If supplied as well as predicted_class, the saliency overlay plot title will *also* be annotated with
        the "name" looked up from the raw predicted_class label.
    confidence : float (Optional)
        If supplied as well as predicted_class, the saliency overlay plot title will also be annotated with
        the confidence score. Should be in 0-1.0 range, will be displayed as percentage.
    cmap : matplotlib.pyplot.colors.ColorMap (Optional)
        A PyPlot colormap to apply for the saliency map. Defaults to plt.cm.plasma
    alpha : float (Optional)
        Opacity of the saliency heatmap to show in the overlay image. Defaults to 0.5
    """
    # Revert image normalization
    image = tensor_to_imgarray(image, floating_point=True)

    # Given the saliency map has already been normalized to 0-1, we can apply pyplot colormap as below:
    # (Otherwise see mpl.colors.Normalize and plt.cm.ScalarMappable(norm=norm, cmap=cmap))
    heatmap = cmap(saliency_map)
    heatmap = heatmap[:, :, :-1]  # Trim off the alpha channel (always 1.0 anyway for typical cmaps)

    # Blend image with heatmap:
    combined_image = alpha * heatmap + (1-alpha) * image

    # Plot
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
    ax0.imshow(image)
    ax0.set_axis_off()
    ax0.set_title("Input image")
    ax1.imshow(combined_image)
    ax1.set_axis_off()
    if predicted_class is None:
        ax1.set_title("Saliency overlay")
    else:
        ax1.set_title("Predicted '{}'{}{}".format(
            str(predicted_class),
            f" ({class_names[predicted_class]})" if class_names is not None else "",
            f", {confidence * 100:.1f}%" if confidence is not None else "",
        ))
    ax2.imshow(heatmap)
    ax2.set_axis_off()
    ax2.set_title("Saliency heatmap")
    plt.tight_layout()
    plt.show()
