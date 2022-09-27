import subprocess
import os
import tensorflow as tf
from concurrent.futures import as_completed
from requests_futures.sessions import FuturesSession

def get_tf_record_filepaths (
    train_file_dir: str, test_file_dir: str,validation_file_dir: str = None, train_valid_split: float = 0.9
) -> dict:
    train_file_paths = tf.io.gfile.glob(f"{train_file_dir}/*.tfrecord")
    test_file_paths = tf.io.gfile.glob(f"{test_file_dir}/*.tfrecord")
    if not validation_file_dir:
        # No validation directory given, split training data instead
        split_ind = int(train_valid_split * len(train_file_paths))
        train_file_paths, validation_file_paths = train_file_paths[:split_ind], train_file_paths[split_ind:]
    else:
        validation_file_paths = tf.io.gfile.glob(f"{validation_file_dir}/*.tfrecord")

    return { "train_files": train_file_paths, "test_files": test_file_paths, "validation_files": validation_file_paths }

def parse_tf_records(tf_records: list, num_records: int = None) -> None:
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
    tf_record_data = run_yt_reqs_async(tf_record_data, yt_id_reqs, session)

def run_yt_reqs_async(tf_record_data, yt_id_reqs, session):
    with FuturesSession(session=session):
        for future in as_completed(yt_id_reqs):
            curr_response = future.result()
            fake_id = curr_response.url[-7:-3]
            if curr_response.status_code == 200:
                yt_id = curr_response.text[10:21]
                video_labels = tf_record_data[fake_id]
                tf_record_data[yt_id] = [video_labels]
                tf_record_data[yt_id].append(download_yt_thumbnail(yt_id))
            del tf_record_data[fake_id]
    return tf_record_data

def create_yt_id_request(fake_id: str, session) -> str:
    '''Grabs YouTube ID from fake-generated id, and returns the request'''
    url_str = f"http://data.yt8m.org/2/j/i/{fake_id[0:2]}/{fake_id}.js"
    req = session.get(url_str)
    return req

def download_yt_thumbnail(yt_id: str) -> None:
    '''Downloads YT Thumbnail, using the YouTubeDL Module, into the download_path'''
    download_path = f'/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/labeled_data/{yt_id}'
    possible_extensions = ['.jpg', '.webp', '.png']
    if not any(os.path.exists(download_path + extension) for extension in possible_extensions):
        yt_url = f"http://www.youtube.com/watch?v={yt_id}"
        try:
            subprocess.Popen(f"youtube-dl --write-thumbnail --skip-download {yt_url} -o {download_path}", shell=True)
        except Exception as error:
            print(f"Error with downloading thumbnail from URL: {yt_url} with error: {error}")
    return download_path
