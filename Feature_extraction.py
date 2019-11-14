# importation des modules de nt_toolbox. 


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

# load code. 
from Loading_Data import * 
#%% SECTION 1 GEOMETRIC ELEMENTS


def get_pixels_mask(mask_) : 
    l = []
    for i in range(mask_.shape[0]) : 
        for j in range(mask_.shape[1]) : 
            if mask_[i][j] == 1 or mask_[i][j] == 255  :
                l.append([i,j])
                
    return l


def get_contours(mask):
    mask = mask.astype('uint8')

    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
    #get the greatest contour : 
    maxi = 0
    cnt = contours[0]
    for k in range(len(contours)) : 
        if len(contours[k]) > maxi : 
            maxi = len(contours[k])
            cnt = contours[k]
    return cnt 

def get_histogramm(im_onechannel,mask) : 
    
    histr = cv2.calcHist([im_onechannel],[0],None,[256],[1,256]) 
    a1 = 0
    
    while histr[a1] <1 : 
        a1 +=1 
    a2 = 0
    while histr[-a2] < 1 : 
            a2 +=1
    size = 256-a1-a2 #on a un paquet de pixel par niveau de gris
    hist = cv2.calcHist([im_onechannel],[0],None, [size],[a1,256-a2])

    return hist, a1, a2


def get_threesold_percentiles(hist,a1,a2,q) : 
    total = np.sum(hist)#q entre 0 et 1
    a = total*q
    hist_sum = 0
    i = 0
    while hist_sum<a and hist_sum<total : 
        hist_sum += hist[i][0]
        i = i+1
    return i
        


def get_number_components(mask) :
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask , 8, cv2.CV_32S)
    
    return num_labels

def get_centroid_moments(mask) : 
    mask = mask.astype('uint8')

    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
    #get the greatest contour : 
    if (len(contours)) > 0 : 
        maxi = 0
        cnt = contours[0]
        for k in range(len(contours)) : 
            if len(contours[k]) > maxi : 
                maxi = len(contours[k])
                cnt = contours[k]
        
        M = cv2.moments(cnt)
        
        if M['m00'] != 0 : 
        
            cy = int(M['m10']/M['m00'])
            cx = int(M['m01']/M['m00'])  
            m = list(M.values())
        else : 
            return 0,0,0
    
        return cx, cy, m
    else : 
        return 0,0,0
    


#%% SECTION II get features
# compute the skewness of the grayscale masked image
        
def skewsym(im_masked_one_channel) :
    res = scipy.stats.skew(im_masked_one_channel.reshape(-1))
    return res



def distance_to_centroid(mask) : 
    cx,cy,m = get_centroid_moments(mask)
    contours = get_contours(mask)
    
    v=[]
    for k in range(len(contours)):
        x = contours[k][0][0]
        y = contours[k][0][1]
        d = np.sqrt((cx-x)**(2) + (cy-y)**(2))
        v.append(d)
    v = v/np.linalg.norm(v)
    av = np.mean(v)
    var = np.var(v)
    
    return(av,var)

def symetry_allegee(mask) : 
    feat = scipy.stats.skew(mask,axis = None)
    return feat
    
def symetry_features(mask) :
    cx,cy,m = get_centroid_moments(mask)
    pixels = get_pixels_mask(mask)
    n = len(pixels)
    s1 = 0
    s2 = 0    
    
    for k in range(n) : 
        i = pixels[k][0]
        j = pixels[k][1]
    
        if i < cx and 2*cx - i < mask.shape[0]: 
            s1 += abs(mask[i][j]-mask[2*cx-i][j])
            
        if j<cy and 2*cy - j < mask.shape[1]: 
            s2 += abs(mask[i][j]-mask[i][2*cy-j])
            
    s1 = s1/n
    s2 = s2/n   
    
    for l in range(10,50,5) : 
        im = scipy.ndimage.interpolation.rotate(mask,l)
        cx,cy,m = get_centroid_moments(im)
        pixels = get_pixels_mask(im)
        n = len(pixels)

        s1_,s2_ = 0,0

        
        for k in range(n) : 
            
            i,j = pixels[k][0],pixels[k][1]
            
            if i < cx and 2*cx -i  < im.shape[0]: 
                s1_ += abs(im[i][j]-im[2*cx-i][j])
            
            if j<cy and 2*cy -j < im.shape[1]: 
                s2_ += abs(im[i][j]-im[i][2*cy-j]) 
                
        s1_ = s1_/n
        s2_ = s2_/n 
        if s1+s2 >s1_+s2_ : 
            s1,s2 = s1_ , s2_

    return s1,s2




#do for all channels, grey, and lab
def asymetry_color_shape(im,mask) : #im on one single channel
    x,y,m = get_centroid_moments(mask)
    v1 = []
    u1 = []
    hist,a1,a2 = get_histogramm(im,mask)
    for k in range(1,9) :
        seuil = get_threesold_percentiles(hist,a1,a2,k*0.1)
        ret, im_s = cv2.threshold(im , seuil, 255,cv2.THRESH_BINARY)

        xt,yt,mt = get_centroid_moments(im_s)
        
        if xt != 0 and yt!= 0 and mt != 0 : 
            d = np.sqrt((x-xt)**(2) + (y-yt)**(2))
            v1.append(d)
            
            d_m =np.sum([(m[i]-mt[i])**(2) for i in range(len(m))])
            u1.append(np.sqrt(d_m))
    
    if np.linalg.norm(u1) != 0 and np.linalg.norm(v1) !=0 : 
        u1 = u1/np.linalg.norm(u1)
        v1 = v1/np.linalg.norm(v1)
    
    nu_u = np.mean(u1)
    var_u = np.var(u1)
    
    nu_v = np.mean(v1)
    var_v = np.var(v1)
    
    return [nu_u,var_u], [nu_v,var_v]

def mean_var(im_onechannel_masked) : 
    mean = np.mean(im_onechannel_masked)
    var = np.var(im_onechannel_masked)
    return(mean,var)
    
def mean_var_contour(im_onechannel_masked,mask) : 
    cnt = get_contours(mask)
    v = []
    for k in range(len(cnt)) : 
        i = cnt[k][0][0]
        j = cnt[k][0][1]
        v.append(im_onechannel_masked[j][i])
    v = v/np.linalg.norm(v)
    
    m = np.mean(v)
    var = np.var(v)
    
    return m,var
        
    
def geometric(im_,mask,l = [25,50,70]) : 
    s=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    im=mph.opening(im_,s)
    hist,a1,a2 = get_histogramm(im,mask)
    
    res = []
    for k in l :
        seuil = get_threesold_percentiles(hist,a1,a2,k*0.01)
        ret, im_s = cv2.threshold(im , seuil, 255,cv2.THRESH_BINARY)
        
        res.append(get_number_components(im_s))
        
    return res
    


#%%
def features(M) : 
    im = masked_img(M)
    im_lab = masked_lab(M)
    mask = load_segmentation(M)
    
    features = []
    
  #  sym = symetry_features(mask)
  #  features.append(sym[0])
  #  features.append(sym[1])
    
    dist = distance_to_centroid(mask)
    features.append(dist[0])
    features.append(dist[1])
    
   
    
  #  geom = geometric(r,mask)
  #  features.append(geom[0])
  #  features.append(geom[1])
  #  features.append(geom[2])
    
    
    
    #blue
    print('bleu')
    b = im[:,:,2]
    
  #  sym_c_s = asymetry_color_shape(b,mask)
  #  features.append(sym_c_s[0][0])
  #  features.append(sym_c_s[0][1])
  #  features.append(sym_c_s[1][0])
  #  features.append(sym_c_s[1][1])
    
    mv = mean_var(b)
    features.append(mv[0])
    features.append(mv[1])
    
    mvc = mean_var_contour(b,mask)
    features.append(mvc[0])
    features.append(mvc[1])
    
    geom = geometric(b,mask)
    features.append(geom[0])
    features.append(geom[1])
    features.append(geom[2])
    
    
    #grey
    print('gris')
    gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    
    mv = mean_var(gr)
    features.append(mv[0])
    features.append(mv[1])
    
    mvc = mean_var_contour(gr,mask)
    features.append(mvc[0])
    features.append(mvc[1])
    
    geom = geometric(gr,mask)
    features.append(geom[0])
    features.append(geom[1])
    features.append(geom[2])
    
    features.append(symetry_allegee(mask))
    features.append(symetry_allegee(im))
    
    
    sym_c_s = asymetry_color_shape(gr,mask)
    features.append(sym_c_s[0][0])
    features.append(sym_c_s[0][1])
    features.append(sym_c_s[1][0])
    features.append(sym_c_s[1][1])
    
    #blue
    mvc = mean_var_contour(b,mask)
    features.append(mvc[0])
    features.append(mvc[1])
    
    
    #grey
    mvc = mean_var_contour(gr,mask)
    features.append(mvc[0])
    features.append(mvc[1])
    
    #luminance
    lum = im_lab [:,:,2]
    mvc = mean_var_contour(lum,mask)
    features.append(mvc[0])
    features.append(mvc[1])
    
    #skew_symetrie
    features.append(skewsym(gr))  
    
    #rouge
    r = im[:,:,0]
    
    
    #vert
    g = im[:,:,1]
    
    
    return features
   

