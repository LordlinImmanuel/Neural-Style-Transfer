import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.applications.vgg19 import VGG19
import numpy as np
import cPickle as pickle

def getWeights():
    tf.enable_eager_execution()
    vgg19=VGG19(include_top=False, weights='imagenet')
    weights={}
    for weight in vgg19.weights:
        weights[weight.name]=weight.numpy()
    
    return weights

with open('data.p', 'w') as fp:
    data=getWeights()
    pickle.dump(data, fp,protocol=pickle.HIGHEST_PROTOCOL)
