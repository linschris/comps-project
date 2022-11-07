import glob
import cv2
import numpy as np

# Constants
DATABASE_PATH = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/predictions"
EVAL_DB_PATH = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/eval/predictions"

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

def grab_images_and_paths(self, image_dir: str, image_paths: list = [], num_images: int = None) -> tuple([list, int]):
    """Grab either a specific number (if num_images is not None) or all the images from a
    directory you specify. 
    
    Note: to ensure we don't get predictions from already-computed images, if the image path already
    exists in the database, we will skip over it.

    Args:
        image_dir (str): The directory to grab images from.
        image_paths (list): The already-predicted (from DB) image-paths (that we can skip over).
        num_images (int): The number of images you want to grab. If None, we grab all images.

    Raises:
        ValueError: If the number of images grabbed is equal to 0, then we throw an Error, as it's
        unlikely you want/expect to grab 0 (for gathering predictions for example).

    Returns:
        tuple(list, int): A tuple containing a list of image paths, and the number of images grabbed.
    """
    loaded_images = []
    loaded_image_paths = []
    curr_image_paths = grab_all_image_paths(image_dir)
    curr_index, num_loaded_images = 0, 0
    while (not num_images or num_loaded_images < num_images) and curr_index < len(curr_image_paths):
        curr_image_path = curr_image_paths[curr_index]
        if curr_image_path not in image_paths:
            try:
                curr_image = get_and_resize_image(curr_image_path, self.model.input_shape[1:])
                loaded_images.append(curr_image)
                loaded_image_paths.append(curr_image_path)
                num_loaded_images += 1
            except Exception as e:
                pass
        curr_index += 1
    if num_loaded_images == 0:
        raise ValueError(f"No images could be found in {image_dir}, or all images from {image_dir} have been grabbed.")
    loaded_images = np.concatenate(loaded_images, axis=0)
    return loaded_image_paths, loaded_images


# Video Utility Functions

def get_youtube_urls(image_scores, k):
    youtube_urls = {}
    for element in image_scores:
        image_path, score = element[0], element[1]
        yt_url, time_str = get_yt_embed_url(image_path)
        if yt_url not in youtube_urls:
            youtube_urls[yt_url] = [] # May change to set
            if len(youtube_urls) >= k:
                break
        if time_str:
            youtube_urls[yt_url].append(time_str)
    return youtube_urls

def get_yt_embed_url(image_path: str) -> tuple([str, int]):
    image_path = image_path.split("/")[-1] # We only want the last part of the path
    video_id = image_path[0:11]
    video_embed_url = f'https://www.youtube.com/embed/{video_id}'
    if len(image_path) > 16: # 11 for YT ID, 5 for .webp (at max), so we're looking at frame
        curr_time_str = int(image_path[-9:-4])
        return video_embed_url, curr_time_str
    return video_embed_url, None

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