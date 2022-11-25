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

### Running the Application

To run the application, run the `run_app.py` python file.
> NOTE: do it from the comps-project folder and not any other subfolder of this project

After loading for a minute or so (due to loading the models and database), a Flask UI should be started
and everything should work!

## Future Work
- Adding more models such as:
    - SIFT / SURF / RootSIFT
    - Ensemble Methods (Combining models like )
    - Inverted Indexing (specifically for the CNN and RMAC models)
- Adding more data
    - Only approximately one million videos compared to the billions on YouTube
    - With better hardware

