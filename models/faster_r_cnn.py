from gluoncv import model_zoo, data
import mxnet as mx
from database import Database

class FasterRCNN():
    ''' Object Detection Model, where the predictions consists of object classes, locations, and their bounding boxes, as opposed to a feature vector.'''

    def __init__(self, database: Database = None):
        self.database = database
        self.model = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
    
    def predict_image_from_path(self, image_path):
        self.predict_image_paths([image_path])
    
    def predict_image_paths(self, image_paths):
        # We cannot simply pass in CV2-imread images, so we must use image paths :/
        object_infos = []
        curr_num = 0
        for image_path in image_paths:
            if image_path[-4:-1] == ".jp" and image_path not in self.database.object_predictions:
                x, orig_img = data.transforms.presets.rcnn.load_test(image_path)
                box_ids, scores, bboxes = self.model(x)
                curr_process = FasterRCNN.parse_object_info(bboxes[0], scores[0],
                                            box_ids[0], class_names=self.model.classes)
                object_infos.append(curr_process)
            curr_num += 1
        return object_infos
    
    def query_image(self, query_img_path, k=5):
        """Queries image based on the matching of objects in this image and database images. In this case, we value the difference in the number of matched objects (we want the same number of relevant objects) and the number of matched object classes (dog, bike, etc.) 

        Args:
            query_img_path (str): String of where the image path is on the file disk
            k (int, optional): The number of top picks to grab (top-k videos). Defaults to 5.

        Returns:
            list[str]: Returns a list of strings, which are the top-k most relevant videos.
        """
        x, query_img = data.transforms.presets.rcnn.load_test(query_img_path)
        box_ids, scores, bboxes = self.model(x)
        query_object_info = FasterRCNN.parse_object_info(bboxes[0], scores[0],
        box_ids[0], class_names=self.model.classes)
        if len(query_object_info) == 0:
            return []
        scores_and_images = {}
        class_names = query_object_info.keys()
        for class_name in class_names:
            if class_name in self.database.object_predictions:
                for image_path in self.database.object_predictions[class_name]:
                    num_img_objects = len(self.database.object_predictions[class_name][image_path])
                    num_query_objects = len(query_object_info[class_name])
                    object_diff_score = 1/(abs(num_img_objects - num_query_objects) + 1) # 1 / diff in the number of objects (2 dogs versus 10 dogs) for this matched object class
                    if image_path not in scores_and_images:
                       scores_and_images[image_path] = 0
                    scores_and_images[image_path] += object_diff_score * sum(self.database.object_predictions[class_name][image_path]) # Add the confidence score of the objects to the overall score of the image, since if it's more confident in it being a particular object, it's likely more relevant (in terms of objects).
        return sorted(scores_and_images.items(), key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def parse_object_info(bboxes, scores=None, labels=None, thresh=0.5,
                    class_names=None):
        '''Parses the info about the objects in the image, which comes from the gluoncv's FasterRCNN model, returning the object classes it sees (which map to an array of confidence scores).'''
        if mx is not None and isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()
        if mx is not None and isinstance(labels, mx.nd.NDArray):
            labels = labels.asnumpy()
        if mx is not None and isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()

        class_ids = {}
        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.flat[i] < thresh:
                continue
            if labels is not None and labels.flat[i] < 0:
                continue
            cls_id = int(labels.flat[i]) if labels is not None else -1
            score = float('{:.3f}'.format(scores.flat[i]))
            if class_names[cls_id] not in class_ids:
                class_ids[class_names[cls_id]] = []
            class_ids[class_names[cls_id]].append(score)
        return class_ids