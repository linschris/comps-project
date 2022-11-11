import subprocess
import os
import tensorflow as tf
from concurrent.futures import Future, as_completed
from requests_futures.sessions import FuturesSession


def parse_tf_records(tf_record_dir: str, download_dir: str, num_records: int = None) -> dict:
    """
    Grabs and parses the TF Records from a given file directory, grabbing the corresponding video labels, and downloading the entire video and thumbnails associated with the video labels.

    Args:
        - tf_record_dir (str): A file directory (folder) containing the tf record file paths.
        - download_dir (str): A directory where the corresponding frames/thumbnails/videos will be downloaded (in videos/thumbnails/frames - subfolder)
        - num_records (int, optional): Integer determining how many TF Records to parse. Defaults to None where all TF Records will be parsed.

    Returns:
        - dict: dictionary mapping YouTube video IDs to arrays containing the thumbnail download path and video download path.
    """
    tf_records = tf.io.gfile.glob(f"{tf_record_dir}/*.tfrecord")
    tf_record_data = {}
    yt_id_reqs = []
    # Allows for YouTube IDs to be fetched asynchronously
    session = FuturesSession(max_workers=8)
    raw_dataset = tf.data.TFRecordDataset(tf_records)
    raw_dataset = raw_dataset.take(num_records) if num_records else raw_dataset
    for raw_record in raw_dataset:
        curr_tensors = tf.io.parse_single_example(
            raw_record,
            features={
                "id": tf.io.FixedLenFeature([1], dtype=tf.string),
                "labels": tf.io.VarLenFeature(dtype=tf.int64)
            }
        )
        fake_video_id = curr_tensors['id'].numpy()[0].decode("utf-8")
        video_labels = list(curr_tensors["labels"].values.numpy())
        tf_record_data[fake_video_id] = video_labels
        yt_id_reqs.append(create_yt_id_request(fake_video_id, session))
    tf_record_data = run_yt_reqs_async(
        tf_record_data, yt_id_reqs, session, download_dir)
    return tf_record_data


def run_yt_reqs_async(tf_record_data: dict, yt_id_reqs: list, session: FuturesSession, download_dir) -> dict:
    """
    Runs YouTube Requests asynchrously, watching for completion of any requests.

    Upon the completion of a YouTube ID Request:
        - The YouTube ID will be parsed from the content of the response.
        - The thumbnail and video download path will be downloaded using the YouTube ID.
        - The map replaces `fake_id -> video_labels` key-value pairs with `YouTube_ID -> [thumbnail, video]` key-value pairs 


    Args:
        - tf_record_data (dict): Key-value pairs of ids-thumbnails
        - yt_id_reqs (list): List containing requests for YouTube ID.
        - session (FuturesSession): The session object allowing persistent (yt-id) requests to occur.
        - download_dir (str): Directory where thumbnails/videos/frames will be downloaded

    Returns:
        - dict: Updated Key-value pairs of yt-ids to thumbnails
    """
    with FuturesSession(session=session):
        for future in as_completed(yt_id_reqs):
            curr_response = future.result()
            fake_id = curr_response.url[-7:-3]
            if curr_response.status_code == 200:
                yt_id = curr_response.text[10:21]
                # video_labels = tf_record_data[fake_id]
                # tf_record_data[yt_id] = [video_labels]
                tf_record_data[yt_id] = download_yt_thumbnail(
                    yt_id, download_dir)
            del tf_record_data[fake_id]
    return tf_record_data


def create_yt_id_request(fake_id: str, session: FuturesSession) -> Future:
    """
    Creates a YouTube ID from the fake ID in the dataset, returning the request to be watched for completion (by the session).

    Args:
        fake_id (str): The fake ID parsed from the TFRecords, 
        session (FuturesSession): The session object allowing persistent (yt-id) requests to occur.

    Returns:
        Future: Asynchrous request (which may or may not be completed) 
    """
    url_str = f"http://data.yt8m.org/2/j/i/{fake_id[0:2]}/{fake_id}.js"
    req = session.get(url_str)
    return req


def download_yt_thumbnail(yt_id: str, download_dir: str) -> list:
    """
    Downloads the YouTube Video and Thumbnail, using the yt-dlp module and YouTube ID, into corresponding video and thumbnails folders in the download directory.

    Args:
        - yt_id (str): The YouTube ID associated with a 'real' YouTube video (can be accessed via `youtube.com/watch/_id_`)
        - download_dir (str): The directory containing the video and thumbnail folders.

    Returns:
        - list: An array containing the download paths to the thumbnail and video respectively.
    """
    possible_extensions = ['.jpg', '.webp', '.png']
    download_thumbnail_path = os.path.join(download_dir, "thumbnails", yt_id)
    download_video_path = os.path.join(download_dir, "videos", yt_id)
    if not any(os.path.exists(download_thumbnail_path + extension) for extension in possible_extensions):
        yt_url = f"http://www.youtube.com/watch?v={yt_id}"
        try:
            # download thumbnail and video
            # --skip-download for thumbnail DL
            # format can be 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
            th = subprocess.Popen(
                f"yt-dlp --write-thumbnail --skip-download {yt_url} -o {download_thumbnail_path}", shell=True)
            v = subprocess.Popen(
                f"yt-dlp {yt_url} -f '160[ext=mp4]' -o {download_video_path}.mp4", shell=True)
        except Exception as error:
            print(f"Error with downloading thumbnail/video from URL: {yt_url} with error: {error}")
    return [download_thumbnail_path, download_video_path]
