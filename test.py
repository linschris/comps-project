import cProfile
import pstats
from database import Database
from data.models.altered_xception import AlteredXception


tf_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/tfrecords/video/train"
download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"
with cProfile.Profile() as pr:
    try:
        db = Database(download_dir)
        xception_model = AlteredXception(db)
        image_paths, loaded_images = xception_model.grab_images_and_paths(
        "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails", 10)
        condensed_fv = xception_model.get_avg_feature_vectors(loaded_images)
        db.store_predictions(condensed_fv, image_paths)
    except KeyboardInterrupt:
        print("User interrupted...getting stats")

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename='needs_profiling.prof')