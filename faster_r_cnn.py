from gluoncv import model_zoo, data
import mxnet as mx
from database import Database
from utils import grab_all_image_paths

class FasterRCNN:
    def __init__(self, db: Database):
        self.database = db
        self.model = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
    
    def __call__(self, query_img_path):
        return self.query_image_by_objects(query_img_path)

    def query_image_by_objects(self, query_img_path, k=5):
        x, query_img = data.transforms.presets.rcnn.load_test(query_img_path)
        box_ids, scores, bboxes = self.model(x)
        query_object_info = get_object_info(bboxes[0], scores[0],
                                            box_ids[0], class_names=self.model.classes)
        if len(query_object_info[0]) <= 0:
            raise ValueError("Invalid image. No object classes could be found.")
        scores_and_images = []
        for image_path, db_obj_info in self.database.object_predictions.items():
            curr_score = 0
            num_category_matches = 0
            if isinstance(db_obj_info["class_names"], dict) and len(db_obj_info["class_names"]) > 0:
                for obj_class, num in db_obj_info["class_names"].items():
                    if obj_class in query_object_info[0]:
                        num_category_matches += 1
                        curr_score += num
            if curr_score > 0:
                scores_and_images.append([image_path, curr_score, num_category_matches])
        return sorted(scores_and_images, key=lambda x: (x[2], x[1]), reverse=True)

def get_object_info(bboxes, scores=None, labels=None, thresh=0.5,
                    class_names=None):
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

def save_object_predictions(image_dir, database, num_images=None):
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
    image_paths = grab_all_image_paths(image_dir)
    object_infos = []
    loaded_image_paths = []
    num_loaded_predictions = 0
    for image_path in image_paths:
        if image_path[-4:-1] == ".jp" and image_path not in database.object_predictions:
            x, orig_img = data.transforms.presets.rcnn.load_test(
                image_path)
            box_ids, scores, bboxes = net(x)
            num_loaded_predictions += 1
            curr_process = get_object_info(bboxes[0], scores[0],
                                           box_ids[0], class_names=net.classes)
            if num_loaded_predictions > num_images:
                break
            object_infos.append(curr_process)
            loaded_image_paths.append(image_path)
    database.store_object_data(object_infos, loaded_image_paths)