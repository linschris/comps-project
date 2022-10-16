import json
import os
import h5py


class Database:
    def __init__(self, database_file_path: str) -> None:
        self.object_predictions_fp = os.path.join(
            database_file_path, 'objects.json')
        self.predictions_fp = os.path.join(
            database_file_path, 'predictions.json')
        self.object_predictions = self.load_json_data(
            self.object_predictions_fp)
        self.predictions = self.load_json_data(self.predictions_fp)

    def store_predictions(self, predictions, image_paths) -> None:
        # Adds new predictions to already stored predictions, overriding stored values for new ones
        image_path_prediction_map = {}
        for index, prediction in enumerate(predictions):
            image_path_prediction_map[image_paths[index]] = prediction.tolist()
        self.predictions.update(image_path_prediction_map)
        with open(self.predictions_fp, 'w') as f:
            f.write(json.dumps(self.predictions) + "\n")

    def store_object_data(self, image_paths, object_infos):
        image_path_object_data_map = {}
        for index, object_info in enumerate(object_infos):
            image_path_object_data_map[image_paths[index]] = {
                "class_names": object_info[0],
                "class_scores": object_info[1],
                "bounding_box_coords": object_info[2],
            }
        self.object_predictions.update(image_path_object_data_map)
        with open(self.object_predictions_fp, 'w') as f:
            f.write(json.dumps(self.object_predictions) + "\n")

    def load_json_data(self, file_path):
        if not os.path.exists(file_path):
            return {}
        with open(file_path) as f:
            try:
                return json.load(f)
            except:
                return {}
