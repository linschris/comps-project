from database import Database
from models.altered_xception import AlteredXception
import cProfile
from models.faster_r_cnn import save_object_predictions
import pstats
# download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"
# db = Database(download_dir)
# xception_model = AlteredXception(db)
# image_paths, loaded_images = xception_model.grab_images_and_paths("/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/frames", 70000)
# image_fvs = xception_model.get_avg_feature_vectors(loaded_images)
# db.store_predictions(image_fvs, image_paths)

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"
        db = Database(download_dir)
        save_object_predictions("/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails", db, 1000)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='needs_profiling.prof')








