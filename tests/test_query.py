import sys
import unittest
sys.path.append('..')
from altered_xception import AlteredXception
from database import Database
from faster_r_cnn import FasterRCNN
from rmac_model import RMACModel
from utils import DATABASE_PATH
from query import query_image


class QueryTestMethods(unittest.TestCase):
    def setUp(self):
        self.database = Database(DATABASE_PATH)
        self.ax_model = AlteredXception(self.database)
        self.rmac_model = RMACModel(self.ax_model.model.get_layer("conv2d_3").output_shape[1:], 3, self.database)
        self.rcnn_model = FasterRCNN(self.database)
        self.num_predictions = 5
        self.query_image_path = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/test-frames/-xqwk75g5c4_frame_00156.jpg"
    
    def test_query(self):
        callable_functions = [self.ax_model, self.rmac_model, self.rcnn_model]
        returned_videos = query_image(self.query_image_path, callable_functions, k=self.num_predictions)
        assert len(returned_videos) == len(callable_functions)
        print(returned_videos)


if __name__ == "__main__":
    unittest.main()