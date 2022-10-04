import cProfile
import pstats
from process_tf_record_data import *

tf_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/tfrecords/video/train"
download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"
with cProfile.Profile() as pr:
    try:
        print(parse_tf_records(tf_dir, download_dir))
    except KeyboardInterrupt:
        print("User interrupted...getting stats")


stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename='needs_profiling.prof')
