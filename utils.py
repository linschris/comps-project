import glob
import cv2
import numpy as np

# Constants
DATABASE_PATH = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/predictions"

def get_and_resize_image(image_path, resize_shape):
    width, height, channels = resize_shape
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (width, height))
    reshaped_img = np.reshape(
        resized_img, [1, width, height, channels]
    )
    return reshaped_img

def grab_all_image_paths(image_dir):
    image_paths = []
    extensions = ('/*.jpg', '/*.webp', '/*.png')
    for extension in extensions:
        image_paths.extend(glob.glob(image_dir + extension))
    return image_paths

# Video Utility Functions

def get_youtube_urls(image_scores):
    youtube_urls = []
    for element in image_scores:
        image_path, score = element[0], element[1]
        yt_url = get_youtube_embed_url(image_path)
        youtube_urls.append(yt_url)
    return youtube_urls

def get_youtube_embed_url(image_path):
    curr_str = (image_path.split("/")[-1]).split("_frame")
    video_id = curr_str[0]
    if len(curr_str) == 2:  # We're looking at a frame
        time_str = int(curr_str[1][1:6])
        return f'https://www.youtube.com/embed/{video_id}?start={time_str}'
    elif "." in video_id:  # We're looking at a thumbnail
        id_without_ext = video_id.split('.')[0]
        return f'https://www.youtube.com/embed/{id_without_ext}'
    else:
        return f'https://www.youtube.com/embed/{video_id}'

def get_youtube_url(image_path):
    curr_str = (image_path.split("/")[-1]).split("_frame")
    video_id = curr_str[0]
    if len(curr_str) == 2:  # We're looking at a frame
        time_str = get_youtube_time_str(curr_str[1][1:6])
        return f'https://www.youtube.com/watch?v={video_id}&t={time_str}'
    else:  # We're looking at a thumbnail
        if "." in video_id:
            id_without_ext = video_id.split('.')[0]
            return f'https://www.youtube.com/watch?v={id_without_ext}'
        return f'https://www.youtube.com/watch?v={video_id}'

def get_youtube_time_str(time_str):
    num_seconds = int(time_str)
    num_hours = int(num_seconds / 3600)
    num_minutes = int(num_seconds / 60)
    num_seconds -= (num_hours * 3600 + num_minutes * 60)
    curr_str = ""
    if num_hours > 0:
        if num_hours < 10:
            curr_str += f'0{num_hours}h'
        else:
            curr_str += f'{num_hours}h'
    if num_minutes > 0:
        if num_minutes < 10:
            curr_str += f'0{num_minutes}m'
        else:
            curr_str += f'{num_minutes}h'
    if num_seconds > 0:
        if num_seconds < 10:
            curr_str += f'0{num_seconds}s'
        else:
            curr_str += f'{num_seconds}s'
    return curr_str

def convert_to_embed_links(yt_urls):
    for i in range(len(yt_urls)):
        yt_urls[i] = yt_urls[i].replace("https://www.youtube.com/watch?v=", "https://www.youtube.com/embed/")
        yt_urls[i] = yt_urls[i].replace("&t=", "?start=")