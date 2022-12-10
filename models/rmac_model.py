from __future__ import division
from __future__ import print_function
from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.models import Model
import keras.backend as K
import sys
sys.path.append('./models')
from database import Database
from roi_pooling import RoiPooling
from altered_xception import AlteredXception
import numpy as np
from utils import get_and_resize_image

class RMACModel:
    def __init__(self, conv_fm_shape: tuple, region_param: int, database: Database = None) -> None:
        '''
            Initializes a RMAC Model based on the output convolutional Feature Map 
            "Shape" which is a tuple containing the height and width of the feature maps, 
            and number of feature maps. 
        '''
        width, height, *num_fm = conv_fm_shape
        self.regions = get_rmac_regions(width, height, region_param)
        self.model = create_rmac_model(len(self.regions)) # Recall it'll have 2 parts
        self.database = database
    
    def predict_image_from_path(self, image_path):
        image = get_and_resize_image(image_path, self.model.input_shape[0][1:])
        return self.predict_images(image)

    def predict_images(self, images):
        '''To predict the images, we resize the images to fit the model's input shape, compute the "description" of the image, and append the feature vector to a list, before converting it to a big numpy array which is returned.'''
        curr_fvs = []
        for image in images:
            image = np.reshape(image, (1, 299, 299, 3))
            curr_fv = self.model.predict([image, np.expand_dims(self.regions, axis=0)])
            curr_fvs.append(curr_fv)
        return np.concatenate(curr_fvs, axis=0)
    
    def query_image(self, query_image_path):
        '''Queries image similar to the AlteredXception Model, by using Euclidean distance and returning the closest (in distance) first in the list.'''
        query_img = get_and_resize_image(query_image_path, self.model.input_shape[0][1:])
        query_fv = self.model.predict([query_img, np.expand_dims(self.regions, axis=0)])
        distances = []
        for image_path in self.database.rmac_prediction_image_paths.keys():
            curr_index = self.database.rmac_prediction_image_paths[image_path]
            curr_fv = self.database.rmac_predictions[curr_index]
            dist = np.linalg.norm(query_fv - curr_fv)
            distances.append([image_path, dist])   
        return sorted(distances, key=lambda x: x[1])

'''RMAC Model Creation Methods'''

def addition(x):
    total = K.sum(x, axis=1)
    return total

def create_rmac_model(num_rois):
    '''
        Not my original code. Refer to https://github.com/noagarcia/keras_rmac.
        Creates an RMAC "Model" which essentially performs the operations from RMAC
        on an input image; this code was readapted for Xception.
        
        It takes in the normal input from the Xception Model as well as the Regions Of Interest.
        After performing ROI-pooling (or essentially computing the region feature vectors), we 
        perform L2-PCA-L2 on the vectors, sum them up to form a single vector, and finally
        normalize this final vector.
        
        Essentially, we've capture many "regions" in a single vector. 
    '''
    
    altered_xception = AlteredXception(output_layer_name='conv2d_4')
    in_roi = Input(shape=(num_rois, 4), name='input_roi')
    
    x = RoiPooling([1, 2, 3, 4], num_rois)([altered_xception.model.output, in_roi])
    # L2 Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)
    # PCA
    x = TimeDistributed(Dense(1024, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)

    # L2 Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)
    # Summation of Vectors into Combined Vector
    rmac = Lambda(addition, output_shape=(1024,), name='rmac')(x)
    # Normalization of Combined Vector
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)
    # Define model
    model = Model([altered_xception.model.input, in_roi], rmac_norm)
    return model

def get_rmac_regions(W, H, L):
    '''
        Not my original code. Refer to https://github.com/noagarcia/keras_rmac.
        
        Determines the regions to look over (i.e. determines square regions which have 40% overlap)
        The width and height, W and H, should be the height and width of the feature maps, coming
        from the last convolutional layer.
        
        The region param L, determines how many, and how large the regions shall be.
    '''
    ovr = 0.4 # desired overlap of neighboring regions
    steps = np.array([2, 3, 4, 5, 6, 7], dtype=np.float) # possible regions for the long dimension
    w = min(W,H)
    b = (max(H,W) - w)/(steps-1)
    idx = np.argmin(abs(((w ** 2 - w*b)/w ** 2)-ovr)) # steps(idx) regions for long dimension
    
    # region overplus per dimension
    Wd, Hd = 0, 0
    if H < W:
        Wd = idx + 1
    elif H > W:
        Hd = idx + 1

    regions = []

    for l in range(1,L+1):

        wl = np.floor(2*w/(l+1))
        wl2 = np.floor(wl/2 - 1)

        b = (W - wl) / (l + Wd - 1)
        if np.isnan(b): # for the first level
            b = 0
        cenW = np.floor(wl2 + np.arange(0,l+Wd)*b) - wl2 # center coordinates
        
        b = (H-wl)/(l+Hd-1)
        if np.isnan(b): # for the first level
            b = 0
        cenH = np.floor(wl2 + np.arange(0,l+Hd)*b) - wl2 # center coordinates

        for i_ in cenH:
            for j_ in cenW:
                # R = np.array([i_, j_, wl, wl], dtype=np.int)
                R = np.array([j_, i_, wl, wl], dtype=np.int)
                if not min(R[2:]):
                    continue

                regions.append(R)

    regions = np.asarray(regions)
    return regions