import json
import os
import h5py


class Database:
    def __init__(self, database_file_path: str) -> None:
        self.database_file_path = database_file_path
        self.predictions = self.load_predictions()

    def store_predictions(self, predictions, image_paths) -> None:
        # Adds new predictions to already stored predictions, overriding stored values for new ones
        image_path_prediction_map = {}
        for index, prediction in enumerate(predictions):
            image_path_prediction_map[image_paths[index]] = prediction.tolist()
        print(len(self.predictions), len(image_path_prediction_map))
        # print(list(image_path_prediction_map.keys())[0] in self.predictions)
        self.predictions.update(image_path_prediction_map)
        with open(self.database_file_path, 'w') as f:
            f.write(json.dumps(self.predictions) + "\n")
            # json.dump(self.predictions, f)

    def load_predictions(self) -> dict:
        if not os.path.exists(self.database_file_path):
            return {}
        with open(self.database_file_path) as f:
            try:
                return json.load(f)
            except:
                return {}

    # def save_predictions(self, predictions, image_paths):
    #     with h5py.File('./data/predictions.h5', 'w') as pred_file:
    #         pred_file.create_dataset("predictions", data=predictions)
    #         pred_file.attrs.create("image_paths", image_paths)

    # def load_predictions(self):
    #     if not os.path.exists('./data/predictions.h5'):
    #         return None, None
    #     with h5py.File('./data/predictions.h5', 'r') as pred_file:
    #         predictions = pred_file['predictions'][:]
    #         image_paths = pred_file.attrs.get("image_paths")
    #         return predictions, image_paths
