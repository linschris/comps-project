from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
import mxnet as mx

from database import Database
import os
from altered_xception import AlteredXception


def get_object_info(img, bboxes, scores=None, labels=None, thresh=0.5,
                    class_names=None, colors=None, ax=None,
                    reverse_rgb=False, absolute_coordinates=True,
                    linewidth=3.5, fontsize=12):

    if mx is not None and isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if mx is not None and isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if mx is not None and isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()

    class_ids = {}
    class_scores = []
    bounding_boxes = []
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if class_names[cls_id] not in class_ids:
            class_ids[class_names[cls_id]] = 0
        class_ids[class_names[cls_id]] += 1
        score = float('{:.3f}'.format(scores.flat[i]))
        class_scores.append(score)
        bounding_boxes.append([int(x) for x in bbox])
    return class_ids, class_scores, bounding_boxes


net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
# im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
#                           'gluoncv/detection/biking.jpg?raw=true',
#                           path='biking.jpg')
# im_fpath = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/frames/-_2onFzLtY8_frame_00014.jpg"
# x, orig_img = data.transforms.presets.rcnn.load_test(im_fpath)
# box_ids, scores, bboxes = net(x)
# # ax = utils.viz.plot_bbox(
# #     orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

# object_info = get_object_info(orig_img, bboxes[0], scores[0],
#                               box_ids[0], class_names=net.classes)
download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"
db = Database(download_dir)
xception_model = AlteredXception(db)
image_paths = xception_model.grab_all_image_paths(
    "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/thumbnails", None)
object_infos = []
for image_path in image_paths:
    if image_path[-4:-1] == ".jp":
        print("HIT")
        x, orig_img = data.transforms.presets.rcnn.load_test(image_path)
        box_ids, scores, bboxes = net(x)
        object_infos.append(get_object_info(orig_img, bboxes[0], scores[0],
                                            box_ids[0], class_names=net.classes))
db.store_object_data(image_paths, object_infos)

# query_img_path = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/biking3.jpg"
# x, query_img = data.transforms.presets.rcnn.load_test(query_img_path)
# box_ids, scores, bboxes = net(x)
# query_object_info = get_object_info(query_img, bboxes[0], scores[0],
#                                     box_ids[0], class_names=net.classes)
# db.store_object_data([query_img_path], [query_object_info])

# query_img_path = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/biking.jpg"
# x, query_img = data.transforms.presets.rcnn.load_test(query_img_path)
# box_ids, scores, bboxes = net(x)
# query_object_info = get_object_info(query_img, bboxes[0], scores[0],
#                                     box_ids[0], class_names=net.classes)
# # db.store_object_data([query_img_path], [query_object_info])

# max_score = 0
# max_img_path = None
# for image_path, db_obj_info in db.object_predictions.items():
#     curr_score = 0
#     num_category_matches = 0
#     # print(db_obj_info["class_names"])
#     if type(db_obj_info["class_names"]) == dict:
#         for obj_class, num in db_obj_info["class_names"].items():
#             print(obj_class, num,
#                   query_object_info[0], obj_class in query_object_info[0])
#             if obj_class in query_object_info[0]:
#                 num_category_matches += 1
#                 curr_score += num * max(num_category_matches * 6 / 5, 1)
#         if curr_score > max_score:
#             print(image_path, db_obj_info["class_names"], curr_score)
#             max_score = curr_score
#             max_img_path = image_path

# print(max_img_path, max_score)
