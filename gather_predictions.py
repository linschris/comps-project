from models.altered_xception import AlteredXception
from models.faster_r_cnn import FasterRCNN
from models.rmac_model import RMACModel
from database import Database
from utils import DATABASE_PATH, EVAL_DB_PATH, grab_images_and_paths
import cProfile
import pstats

# download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"
# db = Database(download_dir)
# xception_model = AlteredXception(db)
# image_paths, loaded_images = xception_model.grab_images_and_paths("/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/frames", 70000)
# image_fvs = xception_model.get_avg_feature_vectors(loaded_images)
# db.store_predictions(image_fvs, image_paths)

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        # database = Database(DATABASE_PATH)
        # a_xception = AlteredXception(database=database)
        # rmac_model = RMACModel(a_xception.model.get_layer("conv2d_3").output_shape[1:], 3, database)
        # image_paths, loaded_images = a_xception.grab_images_and_paths("/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails", 60000)
        # print(len(image_paths), len(loaded_images), image_paths, loaded_images)
        # image_fvs = rmac_model.get_rmac_vectors(loaded_images)
        # database.store_rmac_predictions(image_fvs, image_paths)
        # print(database.rmac_predictions.shape)
        # print(len(database.rmac_prediction_image_paths))
        # save_object_predictions("/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails", db, 1000)
        # thumbnail_image_paths, thumbnail_images = a_xception.grab_images_and_paths("/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/frames", 100000)
        # predictions = a_xception.get_avg_feature_vectors(thumbnail_images)
        # db.store_predictions(predictions, thumbnail_image_paths)
        database = Database(DATABASE_PATH)
        a_xception = AlteredXception(database=database)
        rcnn_model = FasterRCNN(database)
        rmac_model = RMACModel(a_xception.model.get_layer("conv2d_3").output_shape[1:], 3, database)
        cat_frames_path = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/eval/frames"
        thumbnail_path = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails"
        image_paths, loaded_images = grab_images_and_paths(thumbnail_path, database.object_predictions, 1000)

        # xception_predictions = a_xception.get_avg_feature_vectors(loaded_images)
        # rmac_predictions = rmac_model.get_rmac_vectors(loaded_images)
        object_predictions = rcnn_model.get_object_predictions(image_paths)

        # print(len(xception_predictions))
        # print(len(rmac_predictions))
        print(len(object_predictions))
        print(len(image_paths))

        # database.store_predictions(xception_predictions, image_paths)
        # database.store_rmac_predictions(rmac_predictions, image_paths)
        database.store_object_data(object_predictions, image_paths)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='needs_profiling.prof')








