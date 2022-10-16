from altered_xception import AlteredXception
from database import Database
import os
import sklearn
import matplotlib.pyplot as plt
import cProfile
import numpy as np
import cv2
import pstats


tf_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/tfrecords/video/train"
download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"
query_image_path = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails/__ARPL_Q-Ts.jpg"

with cProfile.Profile() as pr:
    db = Database(os.path.join(download_dir, "predictions.json"))
    model = AlteredXception(db)
    image_paths, loaded_images = model.grab_images_and_paths(
        "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails", 10000)
    query_img = model.get_and_resize_image(query_image_path)
    np.append(loaded_images, query_img)

    condensed_fv = model.get_avg_feature_vectors(loaded_images)
    db.store_predictions(condensed_fv, image_paths)
    condensed_query_img_fv = condensed_fv[-1].reshape(1, -1)
    min_dist = float('inf')
    min_image_path = None
    for i in range(0, len(condensed_fv)-1):
        curr_vector = condensed_fv[i].reshape(1, -1)
        curr_image_path = image_paths[i]
        curr_dist = sklearn.metrics.pairwise.euclidean_distances(
            curr_vector, condensed_query_img_fv)
        if curr_dist < min_dist:
            print(min_dist, min_image_path)
            min_dist = curr_dist
            min_image_path = curr_image_path
    plt.imshow(cv2.imread(query_image_path))
    plt.show()
    plt.imshow(cv2.imread(min_image_path))
    plt.show()
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename='needs_profiling.prof')
