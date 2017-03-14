from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import math as m
import numpy as np
from numpy import random
import os
import gzip
import struct
import array
import scipy.special as beta
from scipy.misc import logsumexp
from copy import deepcopy

import autograd.numpy as anp
from autograd.scipy.misc import logsumexp as lse
from autograd.optimizers import sgd, adam
from autograd.util import flatten_func
from autograd import grad
from autograd import elementwise_grad
from autograd.numpy.random import rand, seed

import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)

def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images  = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels  = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels

def my_mnist():
    
    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)
    
    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images  = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels  = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels
    
def load_mnist():
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.ceil(float(N_images) / ims_per_row)
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    return cax

def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)

def binarize_images(images):
    for i in range(images.shape[0]):
        images[i] = np.round(images[i])
        print("Image: ", i)
    return images
    
def get_label(labels, index):
    label = labels[index].nonzero()
    return int(label[0]) 
    
def sort_images(images, labels):
    num_images = images.shape[0]
    len_images = images.shape[1]
    num_labels = labels.shape[1]

    digit_size = int(num_images/num_labels) 
    digit = np.empty(digit_size * len_images).reshape(digit_size, len_images)
    bucket_list = [0]*num_labels

    for i in range(len(bucket_list)):
        bucket_list[i] = digit
        #print("Creating Bucket: ", i)
  
    for i in range(num_images):
        
        label = labels[i].nonzero()
        label = int(label[0])
        temp_image = images[i] 
        bucket_list[label] = np.append(bucket_list[label], [temp_image], axis=0)
        print("Bucket", label, i)
    
    for i in range(len(bucket_list)):
        print(i, bucket_list[i].shape)

    return bucket_list
        
def numerator(theta, images, labels, i):
    pi_c = 0.1
    #print("numerator")
    if type(labels) == int:
        label = labels
    else: 
        label = get_label(labels, i)
    Ber1 = np.power(theta[label], images[i])
    Ber2 = np.power((1-(theta[label])), (1-images[i]))
    Ber = Ber1*Ber2
    return pi_c*Ber #pi_c*np.prod(Ber)
    
    
def print_matrix(mat):
    for c in range(len(mat)):
        for d in range(len(mat[0])):
            print(c, d, mat[c][d])
            
def print_vector(vec):
    for d in range(len(vec)):
        print(d, vec[d])

def log_likelihood(img, lab, w):
    num_img = img.shape[0]  
    
    avg_log = np.array([0.0]*num_img)
    acc = w*0 #np.array([0.0]*num_images)
    count = 0
    p = np.array([0.0]*num_classes)
    
    for i in range(0, num_img):
        sum1 = 0 
        label = get_label(lab, i)
        num = numerator(w, img, label, i)
        den = 0
        #print(i, num)
        for c in range(0, num_classes):
            acc[c] = numerator(w, img, c, i)
            den = den + acc[c]
            
        for c in range(0, num_classes):
            p[c] = logsumexp(acc[c]/den)        
            
        t = np.argmax(p)
        if label == t:
            count = count + 1
        predict_acc = np.round(count/float(.1 + i), 3)
        #avg_log[i] = np.average(np.log(np.abs(num/den)))
        avg_log[i] = logsumexp(num/den)
        
        #print(np.log(np.abs(num/den)))
        sum1 = sum1 + np.log(np.prod(num/den))/num_img     
        #print(i, t, label, count, predict_acc, avg_log[i], num, den)
    avg_log = np.average(avg_log)
    print("Average Log-Likelihood: ", avg_log)    
    print("Predictive Accuracy: ", predict_acc)
    return(avg_log, predict_acc)

def normalize(vec, threshold, rounding = False):
    norm = np.argmax(vec)
    vec = np.abs(vec/vec[norm]-1)
    if rounding == True:
        idx = vec[:] > threshold
        vec[idx] = 1
        idx = vec[:] < threshold
        vec[idx] = 0
        return vec
            #return np.round(vec, 1)
    else:
        return vec
        
        
        
def sigmoid(x):
    t = np.tanh(x)
    s = 0.5*(t+1)
    return s
 #%%

for i in range(5):
    arr = 0
    arr = np.arange(5)  # [0, 1, 2, 3, 4]
    random.seed(2)  # Reset random state
    random.shuffle(arr)  # Shuffle!
    print(arr)

#%%
if __name__ == "__main__":
    
    N_data, main_train_images, main_train_labels, main_test_images, main_test_labels = deepcopy(load_mnist())
    train_images = deepcopy(binarize_images(main_train_images))
    test_images = deepcopy(binarize_images(main_test_images))
    
    #%%
    size = 700   
    sub_train_images = deepcopy(train_images[0:size])
    sub_train_labels = deepcopy(main_train_labels[0:size])
    sub_test_images = deepcopy(test_images[0:size])
    sub_test_labels = deepcopy(main_test_labels[0:size])
   
    num_images = sub_train_images.shape[0]
    num_pixels = sub_train_images.shape[1]
    num_classes = sub_train_labels.shape[1]

    print("SHAPE", sub_train_images.shape)

    
    #%%
    # Question 1B
    label_count = [0] * num_classes #number of images of this class
    N1 = np.zeros((num_classes, num_pixels))
    N0 = np.ones((num_classes, num_pixels))
    N0ones = np.ones((num_classes, num_pixels))
    theta = np.zeros((num_classes, num_pixels))

    for i in range (0, num_images):
        label = get_label(main_train_labels, i)
        label_count[label] = label_count[label] + 1
        N1[label] = N1[label] + train_images[i]
        N0[label] = N0[label] + N0ones[label] - train_images[i]
    theta = (N1 + 1)/ (N0 + N1 + 2)
    #theta = (N1 + 1)/(N1 - N0)
    print("SUM", np.sum(theta))
    plot_images(theta, plt)

#%%
    # Question 1D
    #sum1 = 0
    avg_train, prd_train = log_likelihood(sub_train_images, sub_train_labels, theta)
    avg_test, prd_test = log_likelihood(sub_test_images, sub_test_labels, theta)
    
#%%
"""
    #Question 2C        
    x = sort_images(sub_train_images, sub_train_labels)
    rand_image = theta*0
    rand = 6
    #rand = np.int(np.random.uniform(0,size))
    pi_c = 0.1
    print (rand)
    for i in range(len(x)):
        Ber1 = np.power(theta[i], x[i][rand])
        Ber2 = np.abs(np.power((1-theta[i]), (1-x[i][rand])))
        rand_image[i] = pi_c*Ber1*Ber2
    plot_images(rand_image, plt)
"""                
    
#%%
    #Question 2C     AAAA   
    x = sort_images(sub_train_images, sub_train_labels)
    
 #%% 
    #Question 2Ca     
    rand_image = theta*0
    i=0
    while(i<num_classes):
        rand_class = np.int(np.random.uniform(0, 9))
        bucket_size = x[rand_class].shape[0]
        rand_pic = np.int(np.random.uniform(0,bucket_size))
        img = x[rand_class][rand_pic]
        if np.sum(img) > 0.0:
            rand_image[i] = img
            print(i, rand_class, bucket_size, np.sum(img))
            i = i+1    
    plot_images(rand_image, plt)
#%%    
    #Question 2Cb     
    rand_image = theta*0
    i=0
    while(i<num_classes):
        rand_class = np.int(np.random.uniform(0, 9))
        bucket_size = x[rand_class].shape[0]
        rand_pic = np.int(np.random.uniform(0,bucket_size))
        img = numerator(theta, x[rand_class][rand_pic], rand_class, i)
        img = normalize(img, 0.54, True)
        if np.sum(img) > 0.0:
            rand_image[i] = img
            print(i, rand_class, rand_pic, bucket_size, np.sum(img))
            i = i+1    
    plot_images(rand_image, plt)    
    
#%%
    #Question 2F 
    
    sub_train_images_bottom = deepcopy(sub_train_images[0:size])
    sub_train_images_top = deepcopy(sub_train_images[0:size])
    for i in range(0, size):
        for d in range(0, num_pixels):
            if d < num_pixels/2:
                sub_train_images_bottom[i][d] = 0
            else:
                sub_train_images_top[i][d] = 0
    
    
#%%

    number_pics = 20
    rand_image2 = np.zeros((number_pics, num_pixels))

    for i in range(0, number_pics):
        rand_pic = np.int(np.random.uniform(0, num_images))
        label = get_label(sub_train_labels, rand_pic)
        num_t = numerator(theta, sub_train_images_top, label, rand_pic) 
        den_t = 0
        num_b = 0
        sum1 = 0 
        for c in range(0, num_classes):
            Ber1 = np.exp(theta[c], sub_train_images_bottom[i])
            Ber2 = np.exp((1-theta[c]), (1-sub_train_images_bottom[i]))
            num_b = num_b + (Ber1*Ber2)
            num_b = num_b + Ber1 #THIS IS THE PROB THAT Xd = 1, 
            den_t = den_t + logsumexp(numerator(theta, sub_train_images_top, c, rand_pic))
        sum1 = sum1 + num_b*num_t/den_t
        sum1 = normalize(sum1, True)
        sum1[0:num_pixels/2] = 0
        rand_image2[i] = sub_train_images_top[rand_pic] + sum1
    plot_images(rand_image2, plt)
    

    #%%
    #Question 3D
  
    #im = train_images[0]
    weights = theta*0
    dw = weights
    epochs = 4
    step = 0.0001
    momentum = 0.4
    old_m = 0
    old_dw = 0
    dw = 0
    
#%%    
#==============================================================================
     #Q3D working
     for epo in range(0, epochs):
         for i in range(0, num_images):
             img = sub_train_images[i]
             label = get_label(sub_train_labels, i)
             e = np.exp(weights[label]*img)
             sum_exp = 0
             
             for c in range(0, num_classes):
                 sum_exp = sum_exp + np.exp(weights[c]*img)
             dw = img * (1 - e/sum_exp)
             p = e/sum_exp
             #print(i, label, p[0])
             #s = step*old_dw
             ##s = step*dw
             ##m = momentum*old_m
             ##change = -s + m            
             ##old_dw = dw
             ##old_m = change
             #print("DW: ","Epoch", epo, "|", "Image: ", i, "|",  "Label", label, "|", sum_exp)
             #weights[label] = weights[label] + change
             weights[label] = weights[label] + step*dw
             
     plot_images(weights, plt)
#==============================================================================
    
#%%    
    #Q3D working
    for epo in range(0, epochs):
        for i in range(0, num_images):
            img = sub_train_images[i]
            label = get_label(sub_train_labels, i)
            #e = np.exp(weights[label]*img)
            sum_exp = 0
            num = 0
            den = 0
            
            for c in range(0, num_classes):
                e = np.exp(weights[c]*img)
                num = num + img*e
                den = den + e
            dw = img - num/den
            #p = e/sum_exp
            #print(i, label, p[0])
            #s = step*old_dw
            #s = step*dw
            #m = momentum*old_m
            #change = -s + m            
            #old_dw = dw
            #old_m = change
            #print("DW: ","Epoch", epo, "|", "Image: ", i, "|",  "Label", label, "|", sum_exp)
            #weights[label] = weights[label] + change
            weights[label] = weights[label] + step*dw
            
    plot_images(weights, plt)
    
#%%    
    #Question 3E
    avg_train, prd_train = log_likelihood(sub_train_images, sub_train_labels, weights)
    avg_test, prd_test = log_likelihood(sub_test_images,  sub_test_labels, weights)


#%%

    #Question 4C
       
 def mix(phi, img, num_clusters, mix_prop, max_cluster):
     Ber1 = anp.power(phi, img)
     Ber2 = anp.power(1-phi, 1-img)
     Ber = mix_prop*lse(Ber1*Ber2, axis=1)
     Ber = anp.sum(Ber)
     return anp.log(Ber)
     
#%%
     size = 10000
     epochs = 100
     step = 0.1
     num_clusters = 30
     img = sub_train_images[:size]
     total = size*epochs
     phi = anp.random.normal(0, 0.01, num_clusters*num_pixels).reshape(num_clusters, num_pixels)
     random.seed(10)
     random.shuffle(phi)
     p = 0     
     for i in range(0, num_clusters):
         random.seed(i)
         s = 1/float(i+1)
         phi[i] = anp.random.normal(0, s, num_pixels).reshape(num_pixels)
         #phi[i] = random.shuffle(phi[i])
     random.shuffle(phi)

     #phi = p
     theta  = sigmoid(phi)
     #%%
     for epo in range(0, epochs):
         for i in range(0, img.shape[0]):
             mix_prop = 1/float(num_clusters)
             gradient = grad(mix, argnum = 0)
             p = gradient(phi, img[i], num_clusters, mix_prop, max_cluster)
             phi = phi + step*p
             
             percent = np.round((i +epo*size)/float(total) * 100, 3)
             print("Epoch: ", epo, "Image: ", i, "| Percent Complete: ", percent)
         theta = sigmoid(phi)
         save_images(theta, str(epo))
         anp.save('theta.npy', theta)
         anp.save('phi.npy', phi)

         #%%
    for i in range(0, phi.shape[0]):
        m = np.average(phi[i])
        print("Average", m, i)
        sumx = np.sum(phi[i])
        print("SUM", sumx)
#==============================================================================
###
#==============================================================================
#  Probelm Q4
# #%%
# def mix2(phi, img, num_clusters, mix_prop, max_cluster):
#     Ber1 = anp.power(phi, img)
#     Ber2 = anp.power(1-phi, 1-img)
#     Ber = mix_prop*lse(Ber1*Ber2, axis=1)
#     Ber = anp.sum(Ber, axis=0)
#     return anp.log(Ber)
# #%%
#     size = 100
#     epochs = 100
#     step = 0.05
#     num_clusters = 10
#     img = sub_train_images[0:size]
#     img = np.array([img]*num_clusters)
#     img = img.reshape(img.shape[0], img.shape[2], img.shape[1])
#     total = size*epochs
# 
#     phi = anp.random.normal(0, 0.01, num_clusters*num_pixels*size).reshape(num_clusters, num_pixels, size)
#     theta  = sigmoid(phi)
#     
#     for epo in range(0, epochs):
#         print("Epoch: ", epo)
#         r = theta.sum(axis=1)
#         s = anp.sum(theta)
#         mix_prop = r/s
#         max_cluster = anp.argmax(mix_prop)
#  
#         gradient = elementwise_grad(mix2, argnum = 0)
#         p = gradient(phi, img, num_clusters, mix_prop, max_cluster)
#         phi = phi + step*p
#         
#         theta = sigmoid(phi)
#         d = anp.sum(theta, axis=2)
# 
#         save_images(d, str(epo))
#         anp.save('theta2.npy', theta)
#         anp.save('phi2.npy', phi)
# 
#==============================================================================
