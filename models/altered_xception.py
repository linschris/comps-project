import keras.applications.xception
from keras.models import Model
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from database import Database
from utils import get_and_resize_image
import time

class AlteredXception:
    """An altered model of the pre-trained Xception Model, which has the fully-connected (or more) layers chopped off to extract the feature maps or other outputs."""

    def __init__(self, output_layer_name: str = "avg_pool", random_state: int = 10, database: Database = None) -> None:
        """Initalizes the pre-trained, altered Xception Model.

        Args:
            database (Database): A database instance to query from.
            output_layer_name(str): If you want a specific layer to be the output layer, you can specify its name
            and this layer will become the new output_layer.
            random_state(int): By specifying a seed for predictions, ensure we get consistent (or different) results.

            To get layer names (from any Keras model):
                for layer in keras_model.layers:
                    print(layer.name)
        """
        self.database = database
        self.output_layer_name = output_layer_name
        self.model = self.init_altered_model(output_layer_name)
        self.random_state = random_state
    
    def init_altered_model(self, output_layer_name):
        '''
            Utilizes 'transfer learning' in which we utilize an altered (last layer removed) Xception CNN model trained on ImageNet images as opposed to creating and training our own model.
        '''
        xception_model = keras.applications.xception.Xception()
        xception_model.stop_training = True  # Speeds up prediction process
        last_layer = xception_model.get_layer(output_layer_name)
        last_output = last_layer.output
        new_model = Model(xception_model.input, last_output)
        return new_model

    def predict_image_from_path(self, image_path) -> np.ndarray:
        """Computes and returns a global feature vector from the image.

        Args:
            image_path (str): File path where the image is located on your computer.

        Returns:
            NDArray: A (1, 2048) Numpy Array, acting as a descriptor for the image.
        """
        curr_image = get_and_resize_image(image_path, self.model.input_shape[1:])
        return self.predict_images(curr_image)

    def predict_images(self, images: list[np.ndarray], postprocess: bool = False) -> list[np.ndarray]:
        """Grab `global` feature vectors, or vectors describing the list of images provided.

        Args:
            images (list): A list of image NDArrays
            postprocess (bool, optional): Determines whether to postprocess the vectors using L2 Normalization and PCA. Defaults to False.

        Raises:
            ValueError: If the output layer of our model is not the global average pool layer, then we can't get
            the average feature vectors.

        Returns:
            list[nd.array]: A list of 2048-dimensional vectors, describing the individual images.
        """
        if self.output_layer_name != "avg_pool":
            raise ValueError("Model's output layer is not the average pool layer, please switch the output layer name to be `avg_pool`.")
        avg_feature_vectors = self.model.predict(images)
        if postprocess:
            l2_avg_feature_vectors = preprocessing.normalize(
                avg_feature_vectors, axis=0, norm="l2"
            )
            condensed_vectors = self.pca_feature_vector(l2_avg_feature_vectors)
            l2_condensed_vectors = preprocessing.normalize(
                condensed_vectors, axis=0, norm="l2")
            return l2_condensed_vectors
        return avg_feature_vectors
    
    def pca_feature_vector(self, feature_vectors: np.ndarray):
        """Performs Principal Component Analysis on the list of feature vectors
        describing the images, tries to explain 99% variance for less information loss.

        Args:
            feature_vectors (np.ndarray): List of image feature vectors

        Returns:
            condensed_feature_vectors (np.ndarray): List of condensed feature vectors.
        """
        # random state is like random seed, kept same for predictable results!
        # 0.99 means to grab N components to explain 99% variance (N>=1 refer to # of principal components)
        pca = PCA(n_components=0.99, whiten=True,
                  random_state=self.random_state)
        pca.fit(feature_vectors)
        condensed_feature_vectors = pca.transform(feature_vectors)
        return condensed_feature_vectors
    
    def query_image(self, query_img_path: str) -> list:
        """Queries a given image by the `global` feature vectors to determine
        which images are closest, distance-wise as a similarity metric.

        Args:
            query_img_path (str): File path to the query image.

        Raises:
            ValueError: If we have no database connected to this model, we have no images
            to compare to, so initalize the model with a database.

        Returns:
            list: A list of image paths, sorted by distance, closest to farthest.
        """
        if not isinstance(self.database, Database):
            raise ValueError("Invalid Instance Of Xception. Please initalize this model with a database.")
        query_fv = self.predict_image_from_path(query_img_path)
        t0 = time.time()
        distances = []
        for image_path in self.database.prediction_image_paths.keys():
            curr_index = self.database.prediction_image_paths[image_path]
            curr_fv = self.database.predictions[curr_index]
            dist = np.linalg.norm(query_fv - curr_fv)
            distances.append([image_path, dist])   
        calc = sorted(distances, key=lambda x: x[1])
        t1 = time.time()
        print(t1 - t0)
        return sorted(distances, key=lambda x: x[1])