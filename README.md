# YouTube Image Search
## A Senior Computer Science Comprehensive Project By Christopher Linscott

## What is it?

This is an user interface to allow one to query related YouTube videos (given they exist in the database)
using only images and no text (including metadata from the image). The project uses two models: Object Detection and Convolutional Neural Networks (with an included implementation of RMAC), and showcases results from all of these models to compare and/or see the models' predictions for most related videos.

## How do I get started and run the application?

### Requirements

#### Language: `Python 3.9.1` (Use Python 3.7 <= Python 3.x < Python 3.11)

#### Dependencies: Refer to Step 2 Of Installation Or `requirements.txt` of this repository.

### Hardware / OS

#### OS: `macOS Monterey 12.3.1`

#### Processor: `2.3 GHz 8-Core Intel Core i9`

### Installation

1. Clone the repo

```sh 
    git clone https://github.com/linschris/clustering-algorithms.git
```

2. Install the dependencies
```sh
    pip install -r requirements.txt (Python2)
    pip3 install -r requirements.txt (Python3)
```
### You should be good to go!

### Download tfrecords for (fake) YouTube IDs 

To gather the fake youtube ids, navigate to `data/methods/gather_tf_record_data.py` and run this file. This file contains code to fetch and store tf_records into the `data/tfrecords` directory.

### Gather YouTube videos and thumbnails from the tfrecords

To download the corresponding videos and thumbnails from the tfrecords, navigate to `data/methods/process_tf_record_data.py` which will asynchrously fetch the real YouTube IDs and upon completion of the request, will create a subprocess to download the video and thumbnails, downloading them to a directory of your choice. To run this code, simply call it like 
```python 
from data/methods import process_tf_record_data
def main():
    tf_record_data = parse_tf_records("data/tfrecords", "data/videos", None) # None or specify number of records to parse

    # Do stuff with tf_record_data here
    # ...
```

### Get frames from YouTube videos

To get the frames from the youtube videos, we can utilize useful bash scripts provided by [gsssrao](https://github.com/gsssrao/youtube-8m-videos-frames), one of which is named `generateframesfromvideos.sh`.

To call this, make another folder `frames` to store the frames of the videos in.

Next, navigate (using cd) to the parent of folder containing the videos folder and script file:
> cd data

and run the script to generate the frames from the videos (you can use whatever extension, but I would choose jpg or png):

> bash generateframesfromvideos.sh <path_to_directory_containing_videos> <path_to_directory_to_store_frames> <frames_format>

or:

> bash generateframesfromvideos.sh ./videos ./frames jpg

### Gather and store predictions of these images

Finally, we need to compute descriptions of these images. We can run `gather_predictions.py` to compute predictions for each of our images (it computes 100000 or as many images as you want at a time). Make a folder called `/predictions` or specify the db_path in the argument of `gather_predictions`.

They will be stored in:
- `./predictions/object.json` - RCNN predictions
- `./predictions/predictions.npy` - CNN predictions
- `./predictions/prediction_image_paths.json` - Map from image path to CNN predictions (in npy file array)
- `./predictions/rmac_image_paths.npy` - CNN + RMAC predictions
- `./predictions/prediction_image_paths.json` - Map from image path to CNN + RMAC predictions (in npy file array)

Your database should be good to go!

### Running the Application

To run the application, run the `run_app.py` python file.
> NOTE: do it from the comps-project folder and not any other subfolder of this project

After loading for a minute or so (due to loading the models and database), a Flask UI should be started and everything should work!

### Evaluating the Models

To evaluate the models, you can create a new test folder inside of `eval/test_images`. Using the methods above, you can make a small database inside of `eval/test_images/{insert_test_name}/predictions`. Afterward, find a query image and place inside of it `eval/test_images/{insert_test_name}/query_images`, and determine a ground truth of the top-k most relevant videos (k being whatever you want). Write the URLs of the videos inside of a file called `ground_truth.txt` inside of the test folder.

Finally, walk and run through the steps/cells of `evaluate.ipynb`. The results will show up as outputs of each cell. These results were aggregated different categories of video to create graphs and statistics for my COMPS project to evaluate whether or not it was viable!

## Results


## Future Work
- Adding more models such as:
    - SIFT / SURF / RootSIFT
    - Ensemble Methods (Combining models like )
    - Inverted Indexing (specifically for the CNN and RMAC models)
- Adding more data
    - Only approximately one million videos compared to the billions on YouTube
    - With better hardware

