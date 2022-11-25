import sys
sys.path.append('./')
from models.altered_xception import AlteredXception
from models.faster_r_cnn import FasterRCNN
from models.rmac_model import RMACModel
from database import Database
from utils import DATABASE_PATH, grab_images_and_paths
import cProfile
import pstats

def gather_predictions(image_dir):
    '''Based on available frames or thumbnails, gather/compute predictions (i.e. image representations) to be stored in the database for querying.'''
    database = Database(DATABASE_PATH)
    a_xception = AlteredXception(database=database)
    rmac_model = RMACModel(a_xception.model.get_layer("conv2d_3").output_shape[1:], 3, database)
    rcnn_model = FasterRCNN(database)

    image_paths, loaded_images = grab_images_and_paths(image_dir, [], 1000, 1000)

    # xception_predictions = a_xception.predict_images(loaded_images)
    # rmac_predictions = rmac_model.predict_images(loaded_images)
    object_predictions = rcnn_model.predict_image_paths(image_paths)

    # database.store_predictions(xception_predictions, image_paths)
    # database.store_rmac_predictions(rmac_predictions, image_paths)
    database.store_object_data(object_predictions, image_paths)

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        gather_predictions("/Users/lchris/Desktop/frames")
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='needs_profiling.prof')








