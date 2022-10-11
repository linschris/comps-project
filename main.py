import cProfile
import pstats
from altered_xception import AlteredXception
from process_tf_record_data import *
import json

tf_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/tfrecords/video/train"
download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"
with cProfile.Profile() as pr:
    try:
        # parse_tf_records(tf_dir, download_dir)
        with open(os.path.join(download_dir, 'downloaded2.json'), 'w') as f:
            json.dump(parse_tf_records(tf_dir, download_dir, 100000), f)
        # xception_model = AlteredXception(
        #     "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/frames")
        # xception_model.grab_all_feature_vectors()
        # xception_model.cluster_images()
    except KeyboardInterrupt:
        print("User interrupted...getting stats")


stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename='needs_profiling.prof')
