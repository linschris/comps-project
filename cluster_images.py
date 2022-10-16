from altered_xception import AlteredXception
from database import Database
import os
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import matplotlib.pyplot as plt
import cProfile
import numpy as np
import pstats

tf_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/tfrecords/video/train"
download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"
query_image_path = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails/__ARPL_Q-Ts.jpg"

with cProfile.Profile() as pr:
    db = Database(os.path.join(download_dir, "predictions.json"))
    model = AlteredXception(db)
    # image_paths, loaded_images = model.grab_all_feature_vectors(
    #     "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails", 5000)
    # query_img = model.get_and_resize_image(query_image_path)
    # np.append(loaded_images, query_img)
    # condensed_fv = model.get_avg_feature_vectors(loaded_images)
    # neighbors = NearestNeighbors(n_neighbors=2).fit(condensed_fv)
    # distances, indices = neighbors.kneighbors(condensed_fv)
    # for curr_row in range(0, 10):
    #     print("NEXT ROW")
    #     row = indices[curr_row]
    #     plt.imshow(model.get_and_resize_image(image_paths[row[0]])[0])
    #     plt.show()
    #     for i in range(1, len(row)):
    #         plt.imshow(model.get_and_resize_image(image_paths[row[i]])[0])
    #         plt.show()

    fvs = []
    image_paths = []
    for item in db.predictions.items():
        img_path, fv = item
        if len(fv) == 2048:
            fvs.append(fv)
            image_paths.append(img_path)
    fvs = np.array(fvs)
    # print(fvs.shape)
    l2_avg_feature_vectors = preprocessing.normalize(fvs, axis=0, norm="l2")
    condensed_vectors = model.pca_feature_vector(l2_avg_feature_vectors)
    l2_condensed_vectors = preprocessing.normalize(
        condensed_vectors, axis=0, norm="l2")
    neighbors = NearestNeighbors(n_neighbors=3).fit(l2_condensed_vectors)
    distances, indices = neighbors.kneighbors(l2_condensed_vectors)
    print(distances, indices)
    for curr_row in range(-50, -20):
        print("NEXT ROW", curr_row)
        row = indices[curr_row]
        plt.imshow(model.get_and_resize_image(image_paths[row[0]])[0])
        plt.show()
        for i in range(1, len(row)):
            plt.imshow(model.get_and_resize_image(image_paths[row[i]])[0])
            plt.show()


stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename='needs_profiling.prof')
