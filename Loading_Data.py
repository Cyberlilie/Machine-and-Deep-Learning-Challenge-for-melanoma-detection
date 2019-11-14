# -*- coding: utf-8 -*-
# outils image
from skimage.io import imread
from skimage.transform import resize
from skimage import data
from skimage import filters
from skimage import exposure
from skimage import morphology as mph
from skimage.filters.rank import median
from skimage.morphology import watershed, disk
from skimage.filters import rank
from skimage.filters.rank import minimum
from skimage.util import img_as_ubyte
from skimage import io, color
from skimage.filters import threshold_otsu
import cv2
from scipy import ndimage as ndi
import matplotlib.image as mpimg

#numpy, panda, scipy, pylab
import numpy as np
import pandas as pd
import os
from numpy import array, concatenate,imag, real, cumsum, arange, interp, maximum, transpose, linspace, pi, cos,sin
import scipy
import pylab

#outils d'affichage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.pyplot import plot, axis,subplot,imshow, grid

"""
%pylab inline
%matplotlib inline
%cv2 inline
%load_ext autoreload
%autoreload 2
"""


#%%SECTION I    ######### loading data ######

def masked_img(i) : 
    im = load_segmentation_bool(i)
    image_Segmentation_expand = np.expand_dims(im, axis=2)
    res = (image_Segmentation_expand*load_img(i))
    return res

def masked_lab(i) : 
    im = load_segmentation_bool(i)
    image_Segmentation_expand = np.expand_dims(im, axis=2)
    res = (image_Segmentation_expand*load_img_lab(i))
    return res

def name_im(i) :
    if i < 10 : 
        name_im = 'IM_00000'+ str(i)
    elif i <100 : 
        name_im = 'IM_0000'+ str(i)
    else : 
        name_im = 'IM_000'+ str(i)
    return name_im

def load_img(i) :     
    filename = 'data/im/{}.jpg'.format(name_im(i))
    image = imread(filename)
   # (h,w,c) = image.shape
   # h_div_by_8 = int(h/8)
   # w_div_by_8 = int(w/8)
   # image_downsampled = resize(image,(h_div_by_8,w_div_by_8), mode='reflect')
    return image

def load_blue(i) : 
    f = load_img(i)
    b_cannal = f[:,:,2]
    return b_cannal



def load_img_lab(i) :     
    filename = 'data/im/{}.jpg'.format(name_im(i))
    
    image = color.rgb2lab(imread(filename))
 #   (h,w,c) = image.shape
 #   h_div_by_8 = int(h/8)
 #   w_div_by_8 = int(w/8)
 #   image_downsampled = resize(image,(h_div_by_8,w_div_by_8), mode='reflect')
    return image

def load_segmentation(i) : 
    filename_Segmentation = 'data/im/{}_segmentation.jpg'.format(name_im(i))
    image = imread(filename_Segmentation) # Value 0 or 255
#    (h,w) = image.shape
#    h_div_by_16 = int(h/16)
#    w_div_by_16 = int(w/16)
#    image_downsampled = resize(image,(h_div_by_16,w_div_by_16), mode='reflect')
    return image

def load_segmentation_bool(i) : 
    filename_Segmentation = 'data/im/{}_segmentation.jpg'.format(name_im(i))
    image_Segmentation = imread(filename_Segmentation) # Value 0 or 255
    image = (image_Segmentation/255).astype(np.uint8) # To get uint8
 #   (h,w) = image.shape
 #   h_div_by_8 = int(h/8)
 #   w_div_by_8 = int(w/8)
 #   image_downsampled = resize(image,(h_div_by_8,w_div_by_8), mode='reflect')
    return image
    
