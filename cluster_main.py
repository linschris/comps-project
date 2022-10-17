import cProfile
import pstats
from altered_xception import AlteredXception
from process_tf_record_data import *
from database import Database
import json

tf_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/tfrecords/video/train"
download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"
with cProfile.Profile() as pr:
    try:
        # parse_tf_records(tf_dir, download_dir)
        # with open(os.path.join(download_dir, 'downloaded2.json'), 'w') as f:
        # json.dump(parse_tf_records(tf_dir, download_dir, 100000), f)
        db = Database(os.path.join(download_dir, "predictions.json"))
        xception_model = AlteredXception(db)
        xception_model.cluster_images(
            "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/test-frames")
        # xception_model.cluster_predictions()
    except KeyboardInterrupt:
        print("User interrupted...getting stats")


stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename='needs_profiling.prof')
