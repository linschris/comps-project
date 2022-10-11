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
    image_paths, image_vectors = model.grab_all_feature_vectors(
        "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails", 30)
    query_img_vector = model.grab_image_feature_vector(query_image_path)

    min_dist = float('inf')
    min_image_path = None
    # img_vectors = model.db.predictions
    # i = 0
    for i in range(0, len(image_paths)):
        curr_vector = image_vectors[i]
        curr_image_path = image_paths[i]
        curr_dist = sklearn.metrics.pairwise.euclidean_distances(
            np.array(curr_vector).reshape(1, -1), query_img_vector.reshape(1, -1))
        if curr_dist < min_dist:
            print(min_dist, min_image_path)
            min_dist = curr_dist
            min_image_path = curr_image_path

    # for image_path, curr_vector in img_vectors.items():
    #     curr_dist = sklearn.metrics.pairwise.euclidean_distances(
    #         np.array(curr_vector).reshape(1, -1), query_img_vector.reshape(1, -1))
    #     if curr_dist < min_dist:
    #         print(min_dist, min_image_path)
    #         min_dist = curr_dist
    #         min_image_path = image_path

    plt.imshow(cv2.imread(query_image_path))
    plt.show()
    plt.imshow(cv2.imread(min_image_path))
    plt.show()
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename='needs_profiling.prof')
