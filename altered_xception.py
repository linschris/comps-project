import keras.applications.xception
from keras.models import Model
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from database import Database
from utils import get_and_resize_image, grab_all_image_paths

class AlteredXception:
    """An altered model of the Xception Model, mainly for transfer learning."""

    def __init__(self, database: Database = None, output_layer_name: str = "avg_pool", random_state: int = 10) -> None:
        """Initalizes the pre-trained, altered Xception Model.

        Args:
            database (Database): A database instance to store predictions to.
            output_layer_name(str): If you want a specific layer to be the output layer, you can specify its name
            and this layer will become the new output_layer.
            random_state(int): By specifying a seed for predictions, ensure we get consistent (or different) results.

            To get layer names (from any Keras model):
                for layer in keras_model.layers:
                    print(layer.name)
        """
        self.db = database
        self.output_layer_name = output_layer_name
        self.model = self.init_altered_model(output_layer_name)
        self.random_state = random_state

    def __call__(self, query_img_path) -> any:
        return self.query_img_by_global_feature_vector(query_img_path)

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

    def get_image_feature_vector(self, image_path) -> np.ndarray:
        """Computes and returns a global feature vector from the image.

        Args:
            image_path (str): File path where the image is located on your computer.

        Returns:
            NDArray: A (1, 2048) Numpy Array, acting as a descriptor for the image.
        """
        curr_image = get_and_resize_image(image_path, self.model.input_shape[1:])
        return self.get_avg_feature_vectors(curr_image)

    def grab_images_and_paths(self, image_dir: str, num_images: int = None) -> tuple([list, int]):
        """Grab either a specific number (if num_images is not None) or all the images from a
        directory you specify. 
        
        Note: to ensure we don't get predictions from already-computed images, if the image path already
        exists in the database, we will skip over it.

        Args:
            image_dir (str): The directory to grab images from.
            num_images (int): The number of images you want to grab. If None, we grab all images.

        Raises:
            ValueError: If the number of images grabbed is equal to 0, then we throw an Error, as it's
            unlikely you want/expect to grab 0 (for gathering predictions for example).

        Returns:
            tuple(list, int): A tuple containing a list of image paths, and the number of images grabbed.
        """
        loaded_images = []
        loaded_image_paths = []
        curr_index, num_loaded_images = 0, 0
        curr_image_paths = grab_all_image_paths(image_dir)
        while (not num_images or num_loaded_images < num_images) and curr_index < len(curr_image_paths):
            curr_image_path = curr_image_paths[curr_index]
            if curr_image_path not in self.db.prediction_image_paths:
                try:
                    curr_image = get_and_resize_image(curr_image_path, self.model.input_shape[1:])
                    loaded_images.append(curr_image)
                    loaded_image_paths.append(curr_image_path)
                    num_loaded_images += 1
                except Exception:
                    pass
                curr_index += 1
            else:
                curr_image_paths.pop(curr_index)
        if num_loaded_images <= 0:
            raise ValueError(f"No images could be found in {image_dir}, or all images from {image_dir} have been grabbed.")
        loaded_images = np.concatenate(loaded_images, axis=0)
        return loaded_image_paths, loaded_images

    def get_avg_feature_vectors(self, images: list[np.ndarray], postprocess: bool = False) -> list[np.ndarray]:
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
        """Compute Principal Component Analysis on the list of feature vectors
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
    
    def query_img_by_global_feature_vector(self, query_img_path: str) -> list:
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
        if not isinstance(self.db, Database):
            raise ValueError("Invalid Instance Of Xception. Please initalize this model with a database.")
        query_fv = self.get_image_feature_vector(query_img_path)
        distances = []
        for image_path in self.db.prediction_image_paths.keys():
            curr_index = self.db.prediction_image_paths[image_path]
            curr_fv = self.db.predictions[curr_index]
            dist = np.linalg.norm(query_fv - curr_fv)
            distances.append([image_path, dist])   
        return sorted(distances, key=lambda x: x[1])
