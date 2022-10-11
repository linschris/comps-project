import cProfile
import os
import pstats
import keras.applications.xception
from keras.models import Model
import cv2
import numpy as np
import h5py
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm
import glob
from process_tf_record_data import *
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering


class AlteredXception:
    """An altered model of the Xception Model, mainly for transfer learning."""

    def __init__(self, database, region_param=1, output_layer_name="avg_pool") -> None:
        """Initalizes the pre-trained, altered Xception Model.

        Args:
            image_dir (str): A string of the path to the directory containing your image(s).
            region_param (int, optional): Parameter for determining the regions to grab feature vectors from. Defaults to 1, but 1 <= region_param <= inf.
        """
        self.db = database
        self.model = self.init_altered_model(output_layer_name)
        self.region_param = region_param

    def init_altered_model(self, output_layer_name):
        '''
            Utilizes 'transfer learning' in which we utilize an altered (last layer removed) Xception CNN model trained on ImageNet images as opposed to creating and training our own model.

            To get layer names:
            ```
                for layer in xception_model.layers:
                    print(layer.name)
            ```
        '''
        xception_model = keras.applications.xception.Xception()
        xception_model.stop_training = True  # Speeds up prediction process
        last_layer = xception_model.get_layer("avg_pool")
        last_output = last_layer.output
        new_model = Model(xception_model.input, last_output)
        return new_model

    def get_and_resize_image(self, image_path):
        num_samples, width, height, channels = self.model.input_shape
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, (width, height))
        reshaped_img = np.reshape(
            resized_img, [1, width, height, channels])
        return reshaped_img

    def grab_image_feature_vector(self, image_path):
        curr_image = self.get_and_resize_image(image_path)
        return self.get_avg_feature_vectors(curr_image)

    def grab_all_feature_vectors(self, image_dir, num_images=2500):
        loaded_images = []
        curr_index, num_loaded_images = 0, 0
        curr_image_paths = glob.glob(image_dir + "/*")
        while num_loaded_images < num_images and curr_index < len(curr_image_paths):
            curr_image_path = curr_image_paths[curr_index]
            # if curr_image_path not in self.db.predictions:
            try:
                curr_image = self.get_and_resize_image(curr_image_path)
                loaded_images.append(curr_image)
                num_loaded_images += 1
            except:
                if os.path.exists(curr_image_path):
                    os.remove(curr_image_path)
            curr_index += 1
        if num_loaded_images > 0:
            loaded_images = np.concatenate(loaded_images, axis=0)
            feature_vectors = self.get_avg_feature_vectors(loaded_images)
            # self.db.store_predictions(feature_vectors, curr_image_paths)
            return curr_image_paths, feature_vectors  # TODO: fix here
        return []

    def get_avg_feature_vectors(self, images):
        # NOTE: this function should be used if the final layer in the Xception model is the avg. pool layer! Otherwise, use get_image_feature_maps()
        avg_feature_vectors = self.model.predict(images)
        # avg_feature_vectors = self.l2_normalize_vectors(avg_feature_vectors)
        # condensed_vectors = self.pca_feature_vector(avg_feature_vectors)
        # condensed_vectors = self.l2_normalize_vectors(condensed_vectors)
        return avg_feature_vectors

    def cluster_images(self):
        loaded_predictions = self.load_predictions()
        condensed_fv = self.grab_all_feature_vectors(
        ) if loaded_predictions is None else loaded_predictions
        print(loaded_predictions)
        spectral = KMeans(n_clusters=10)
        # spectral = SpectralClustering(
        #     n_clusters=100, n_components=512, assign_labels="discretize")
        spectral.fit(condensed_fv)
        groups = {}
        for index in range(self.num_images):
            image_path = self.image_paths[index]
            cluster = spectral.labels_[index]
            if cluster not in groups:
                groups[cluster] = []
            groups[cluster].append(image_path)
        self.plot_groups(groups)

    def plot_groups(self, groups):
        plt.figure(figsize=(25, 25))
        for cluster in groups:
            print(cluster, groups[cluster])
            files = groups[cluster]
            # only allow up to 30 images to be shown at a time
            if len(files) > 30:
                print(f"Clipping cluster size from {len(files)} to 30")
                files = files[:29]
            # plot each image in the cluster
            for index, file in enumerate(files):
                plt.subplot(10, 10, index+1)
                img = cv2.imread(file)
                img = np.array(img)
                plt.imshow(img)
                plt.axis('off')
            plt.show()

    def get_max_feature_vectors(self, images):
        '''Grabs K feature maps from output of CNN per image, where K is the number of total filters applied on the image.'''
        feature_maps = self.model.predict(images)
        max_feature_vectors = []
        for image in images:
            curr_feature_map = self.model.predict(image)
            # TODO: reshape so its shape is (channels, height, width)
            max_feature_vectors.append(
                self.construct_feature_vector(curr_feature_map))
        return max_feature_vectors

    def construct_feature_vector(self, feature_maps):
        '''Constructs a feature vector of size K (i.e. number of channels) by grabbing the max value on each feature map 1, ..., k.'''
        num_channels = feature_maps.shape[0]
        feature_vector = []
        for channel in range(num_channels):
            feature_vector.append(np.max(feature_maps[channel]))
        return feature_vector

    def pca_feature_vector(self, feature_vectors):
        # random state is like random seed, kept same for predictable results!
        pca = PCA(n_components=0.99, whiten=True, random_state=10)
        print(feature_vectors.shape)
        pca.fit(feature_vectors)
        condensed_feature_vector = pca.transform(feature_vectors)
        # print(feature_vectors.shape)
        print(condensed_feature_vector.shape)
        return condensed_feature_vector

    def get_regions(self):
        # Get RMAC regions based on l i.e. scale parameter
        square_size = 0
        width, height, channels = cv2.resize(
            cv2.imread(self.image_path), (299, 299)).shape

        if self.region_param == 1:
            square_size = min(width, height)
        else:
            square_size = 2*min(width, height) / (self.region_param + 1)
        print(square_size, square_size)

    @staticmethod
    def cosine_similarity(fv1, fv2):
        '''Computes the cosine similarity between two feature vectors, fv1 and fv2.'''
        return dot(fv1, fv2) / (norm(fv1) * norm(fv2))

    def l2_normalize_vectors(self, image_vectors):
        return preprocessing.normalize(image_vectors, axis=0, norm="l2")

    def l2_normalize(self, feature_map_arr):
        feature_map = feature_map_arr[0]
        width, height, channels = feature_map.shape
        print(feature_map.shape)
        for channel in range(0, channels):
            print(feature_map[channel].shape)
            normalized_feature_map = preprocessing.normalize(
                feature_map[channel], norm="l2")
            print(normalized_feature_map)
        return normalized_feature_map


# model = AlteredXception("/Users/lchris/Desktop/labeled_data",
#                         output_layer_name="block14_sepconv2")
# with cProfile.Profile() as pr:
#     model.cluster_images()

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.dump_stats(filename='needs_profiling.prof')
