import subprocess
import requests, os, cv2, random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from requests_futures.sessions import FuturesSession
from concurrent.futures import as_completed


def grab_tf_record_filepaths(train_file_dir: str, test_file_dir: str, validation_file_dir: str = None, train_valid_split: float = 0.9) -> dict:
    train_file_paths = tf.io.gfile.glob(f"{train_file_dir}/*.tfrecord")
    test_file_paths = tf.io.gfile.glob(f"{test_file_dir}/*.tfrecord")
    if not validation_file_dir:
        # No validation directory given, split training data instead
        split_ind = int(train_valid_split * len(train_file_paths)) 
        train_file_paths, validation_file_paths = train_file_paths[:split_ind], train_file_paths[split_ind:]
    else:
        validation_file_paths = tf.io.gfile.glob(f"{validation_file_dir}/*.tfrecord")

    return { "train_files": train_file_paths, "test_files": test_file_paths, "validation_files": validation_file_paths }

def parse_tf_records(tf_records: list, num_records: int = None) -> tuple:
    tf_record_data = {}
    yt_id_reqs = []
    session = FuturesSession(max_workers=8) # Allows for YouTube IDs to be fetched asynchronously
    raw_dataset = tf.data.TFRecordDataset(tf_records)
    raw_dataset = raw_dataset.take(num_records) if num_records else raw_dataset

    for raw_record in raw_dataset:
        curr_tensors = tf.io.parse_single_example(
            raw_record, 
            features = {
                "id": tf.io.FixedLenFeature([1], dtype=tf.string),
                "labels": tf.io.VarLenFeature(dtype=tf.int64)
            }
        )
        fake_video_id = curr_tensors['id'].numpy()[0].decode("utf-8")
        video_labels = list(curr_tensors["labels"].values.numpy())
        tf_record_data[fake_video_id] = video_labels
        yt_id_reqs.append(create_yt_id_request(fake_video_id, session))
    
    with FuturesSession(session=session) as sess:
        for future in as_completed(yt_id_reqs):
            curr_response = future.result()
            fake_id = curr_response.url[-7:-3]
            if (curr_response.status_code == 200):
                yt_id = curr_response.text[10:21]
                video_labels = tf_record_data[fake_id]
                tf_record_data[yt_id] = [video_labels]
                tf_record_data[yt_id].append(download_yt_thumbnail(yt_id))
            del tf_record_data[fake_id]

    return create_cnn_datapoints(tf_record_data)
    
def create_yt_id_request(fake_id: str, session) -> str:
    '''Grabs YouTube ID from fake-generated id, and returns the request'''
    url_str = f"http://data.yt8m.org/2/j/i/{fake_id[0:2]}/{fake_id}.js"
    r = session.get(url_str)
    return r

def download_yt_thumbnail(yt_id: str) -> None:
    download_path = f'/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/labeled_data/{yt_id}'
    possible_extensions = ['.jpg', '.webp', '.png']
    if not any(os.path.exists(download_path + extension) for extension in possible_extensions):
        yt_url = f"http://www.youtube.com/watch?v={yt_id}"
        try:
            subprocess.Popen(f"youtube-dl --write-thumbnail --skip-download {yt_url} -o {download_path}", shell=True)
        except Exception as e:
            print(f"Error with downloading thumbnail from URL: {yt_url} with error: {e}")
    return download_path

def create_labels_array(labels):
    '''Convert numbered labels to array of size 3862, with the indices (of the numbers) as 1s'''
    new_labels = [[0] * 3862 for i in range(len(labels))]
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            new_labels[i][labels[i][j]] = 1
    return new_labels

def create_cnn_datapoints(tf_data: dict, img_size: int = 256) -> tuple:
    ''' 
        Creates a tuple of the YouTube thumbnail img array and corresponding human labels, which
        will act as a datapoint for the CNN to learn from. 
    '''
    x, y = [], []
    possible_extensions = [".jpg", ".webp", ".png"]
    img_array = None
    for data in tf_data.values():
        labels, img_path = data[0], data[1]
        for extension in possible_extensions:
            # As we don't know what type of image youtube-dl downloads, we'll 
            # test possible extensions and break if it works
            try:
                img_array = cv2.imread(img_path + extension)
                resized_img = cv2.resize(img_array, (img_size, img_size))
                break
            except:
                continue
        if img_array is not None:
            y.append(labels)
            x.append(resized_img)
    x = np.array(x).reshape(-1, img_size, img_size, 3)
    
    # Normalize image data
    x = x.astype('float32')
    x /= 255

    y = np.array(create_labels_array(y))
    return x, y

def main():
    pass
    # filepaths = grab_and_split_data(
    #     "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/video/train",
    #     "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/video/test",
    #     "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/video/validate"
    # )
    # print(filepaths)
    # view_tf_records(filepaths["validation_files"], 2)
    # download_yt_thumbnail("dg45mfgd4")
    # load_data("./labeled_data")



if __name__ == "__main__":
    main()
