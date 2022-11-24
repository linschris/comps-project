from bs4 import BeautifulSoup
import requests
import os

def get_tf_records(eval_type: str, data_type: str, tf_record_dir: str) -> None:
    """Get YouTube8M Data, in the form of TFRecords from their Google API Storage.
    Writes them to the specified directory, creating a subfolder based on both the evalulation and data type.

    e.g. `./tfrecords/video/test/*.tfrecord` for `./tfrecords` tf_record_dir, `video` data_type, `test` eval_type

    Args:
        eval_type (str): The type of data, whether you want the training ("train"), testing ("test"), and/or validation ("validation") data.
        data_type (str): Either the video features (video), frame features (frame), or segment features (segment).
    """
    r = requests.get(
        f"http://storage.googleapis.com/us.data.yt8m.org/2/{data_type}/{eval_type}/index.html")
    data = r.text
    soup = BeautifulSoup(data)
    for link in soup.find_all('a'):
        tf_record_full_name = link.get('href')
        tf_record_name = tf_record_full_name.split(".")[0]
        if not os.path.exists(f"./tfrecords/{data_type}/{eval_type}/{tf_record_name}.tfrecord"):
            print(f"Grabbing {tf_record_name}.tfrecord")
            curr_response = requests.get(
                f"http://storage.googleapis.com/us.data.yt8m.org/2/{data_type}/{eval_type}/{tf_record_name}.tfrecord")
            with open(f"{tf_record_dir}/{data_type}/{eval_type}/{tf_record_name}.tfrecord", "wb") as f:
                f.write(curr_response.content)

if __name__ == "__main__":
    eval_type = "test"  # train, test, validate
    data_type = "video"  # video, frame, segment
    tf_record_dir = "../tfrecords"
    get_tf_records(eval_type, data_type, tf_record_dir)