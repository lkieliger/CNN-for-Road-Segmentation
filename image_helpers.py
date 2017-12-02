from program_constants import *
import numpy

def img_crop(im, w, h):
    """
    Extract patches from an image
    
    :param im: The image from which to extract the patches
    :param w: The patches width
    :param h: The patches height
    :return: A list containing all the extracted patches in column order
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


def label_to_img(imgwidth, imgheight, w, h, labels):
    """
    Convert an array of class labels to an image
    
    :param imgwidth: The width of the resulting image
    :param imgheight: The height of the resulting image
    :param w: The width of each color patch that form the image
    :param h: The height of each color patch that form the image
    :param labels: The class labels
    
    :return: A two dimensional image made of several patches 
    with either 1 or 0 as a value, depending on the corresponding label
    """
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j + w, i:i + h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    """
    Convert an image where each pixel is specified as a serie of floats
    of depth PIXEL_DEPTH into an 8-bit image. The range of float is automatically
    mapped to the full 8-bit range of 0-255.
    
    :param img: The image to be converted
    
    :return: The converted image
    """
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def concatenate_images(img, gt_img):
    """
    Concatenate an RGB image with a ground truth binary image. The funciton assumes that the input
    images store float values.
    
    :param img: The RGB (original) image
    :param gt_img: The ground truth binary image, converted to 3 channels if monochrome
    
    :return: The result of the concatenation of the two images.
    """

    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]

    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)

    return cimg


def make_img_overlay(img, predicted_img):
    """
    Overlay the predicted class labels for each image patch on top of the original
    RGB image using transparency.
    
    :param img: The original RGB image, storing float values
    :param predicted_img: The image reconstructed from predictions
    
    :return: An 8-bit RGBA image resulting of the blending of img and predicted img
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img