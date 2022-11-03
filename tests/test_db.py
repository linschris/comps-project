import sys
sys.path.append('..')
import unittest
import numpy
from altered_xception import AlteredXception
from database import Database


class DBTestMethods(unittest.TestCase):
    def setUp(self):
        data_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/predictions"
        self.database = Database(data_dir)
        self.model = AlteredXception(self.database)
        self.num_predictions = 10
        self.image_paths, self.test_predictions = self.get_test_predictions()
    
    def get_test_predictions(self):
        image_paths, loaded_images = self.model.grab_images_and_paths(
        "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails", self.num_predictions)
        condensed_fv = self.model.get_avg_feature_vectors(loaded_images)
        return image_paths, condensed_fv
    
    def test_store_predictions(self):
        '''Number of new predictions should increase by the number of predictions given to grab/store, in the test.'''
        prev_predictions_size = numpy.size(self.database.predictions, axis=0) or 0
        prev_image_path_size = len(self.database.prediction_image_paths)
        self.assertEqual(prev_predictions_size, prev_image_path_size, "Number of stored predictions is not equal to the number of stored image paths.")
        
        self.database.store_predictions(self.test_predictions, self.image_paths)
        new_predictions = self.database.load_npy_data(self.database.predictions_fp)
        new_image_paths = self.database.load_json_data(self.database.prediction_image_paths_fp)
        self.assertTrue(type(new_predictions) == numpy.ndarray, "Predictions is of type NDArray")
        self.assertEqual(numpy.size(new_predictions, axis=0), len(new_image_paths), "Both new predictions and new image paths have been added.")
        self.assertEqual(numpy.size(new_predictions, axis=0), prev_image_path_size + self.num_predictions, "The number of new predictions and image paths corresponds with the number of predictions requested.")

if __name__ == "__main__":
    unittest.main()