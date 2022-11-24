import json
import os
import numpy
import ujson

class Database:
    """
        A fake database based on JSON and NPY files to store the predictions the models make, which then
        can be queried against.
    """
    def __init__(self, database_file_path: str) -> None:
        # Filepaths to load data from
        self.object_predictions_fp = os.path.join(database_file_path, 'objects.json')
        self.predictions_fp = os.path.join(database_file_path, 'predictions.npy')
        self.prediction_image_paths_fp = os.path.join(database_file_path, 'prediction_image_paths.json')
        self.rmac_predictions_fp = os.path.join(database_file_path, 'rmac_preds.npy')
        self.rmac_prediction_image_paths_fp = os.path.join(database_file_path, 'rmac_image_paths.json')
        
        # Loading initial data (if any exists)
        self.object_predictions = self.load_json_data(self.object_predictions_fp)
        self.prediction_image_paths = self.load_json_data(self.prediction_image_paths_fp)
        self.predictions = self.load_npy_data(self.predictions_fp)
        self.rmac_predictions = self.load_npy_data(self.rmac_predictions_fp)
        self.rmac_prediction_image_paths = self.load_json_data(self.rmac_prediction_image_paths_fp)

    def store_predictions(self, predictions, image_paths) -> None:
        # Adds new predictions to already stored predictions, or creates new file to store predictions
        if numpy.size(self.predictions, axis=0) > 0:
            self.predictions = numpy.append(self.predictions, predictions, axis=0)
        else:
            self.predictions = predictions
        numpy.save(self.predictions_fp, self.predictions)
        
        image_path_prediction_map = {}
        max_index = len(self.prediction_image_paths)
        for i in range(max_index, max_index + len(image_paths)):
            image_path_prediction_map[image_paths[i - max_index]] = i
        self.prediction_image_paths.update(image_path_prediction_map)

        self.store_json_data(self.prediction_image_paths, self.prediction_image_paths_fp)
    
    def store_rmac_predictions(self, predictions, image_paths) -> None:
        # This is essentially repeated code, but I decided this over a flag to determine
        # where the predictions we provide should be stored.
        
        if numpy.size(self.rmac_predictions, axis=0) > 0:
            self.rmac_predictions = numpy.append(self.rmac_predictions, predictions, axis=0)
        else:
            self.rmac_predictions = predictions
        numpy.save(self.rmac_predictions_fp, self.rmac_predictions)
        
        image_path_prediction_map = {}
        max_index = len(self.rmac_prediction_image_paths)

        for i in range(max_index, max_index + len(image_paths)):
            image_path_prediction_map[image_paths[i - max_index]] = i
        self.rmac_prediction_image_paths.update(image_path_prediction_map)
        
        self.store_json_data(self.rmac_prediction_image_paths, self.rmac_prediction_image_paths_fp)
        
    def store_object_data(self, object_infos, image_paths):
        image_path_object_data_map = {}
        for index, object_info in enumerate(object_infos):
            image_path_object_data_map[image_paths[index]] = {
                "class_names": object_info[0],
                "class_scores": object_info[1],
                "bounding_box_coords": object_info[2],
            }
        self.object_predictions.update(image_path_object_data_map)
        self.store_json_data(self.object_predictions_fp, self.object_predictions)
    
    def store_json_data(self, json_data, json_file_path):
        with open(json_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(json_data) + "\n")
    
    def load_json_data(self, json_file_path):
        if not os.path.exists(json_file_path):
            return {}
        with open(json_file_path, encoding='utf-8') as json_file:
            try:
                return ujson.load(json_file) # uJSON is faster for loading large JSON files
            except Exception:
                return {}
    
    def load_npy_data(self, npy_file_path):
        if not os.path.exists(npy_file_path):
            return numpy.empty((0, 0))
        try:
            return numpy.load(npy_file_path)
        except Exception:
            return numpy.empty((0, 0))