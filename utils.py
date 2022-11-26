import glob
import cv2
import numpy as np
import os

# Constants
DATABASE_PATH = os.path.join(os.getcwd(), "data", "predictions")
EVAL_DB_PATH = os.path.join(os.getcwd(), "eval", "predictions")
DOG_DB_PATH = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/eval/dog-predictions"
CAT_DB_PATH = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/eval/cat-predictions"

def get_and_resize_image(image_path: str, resize_shape: tuple) -> np.ndarray:
    """Load and resize a image to match a input shape, normally a model input shape.

    Args:
        image_path (str): Path to image to be loaded and resized.
        resize_shape (Tuple): Tuple of size n dimensions denoting each dimensions size.

    Returns:
        np.ndarray: Image as a resized NDArray, the same shape as the resize shape
    """
    width, height, channels = resize_shape
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (width, height))
    reshaped_img = np.reshape(
        resized_img, [1, width, height, channels]
    )
    return reshaped_img

def grab_all_image_paths(image_dir: str, extensions: list[str] = ['/*.*']) -> list[str]:
    """Grab all the image paths of images with extensions (jpg, webp, and png) from a given directory.

    Args:
        image_dir (str): Directory to pull images from.

    Returns:
        list[str]: List of image paths as strings.
    """
    image_paths = []
    for extension in extensions:
        image_paths.extend(glob.glob(image_dir + extension))
    return image_paths

def grab_images_and_paths(image_dir: str, image_paths: list[str] = [], num_images: int = 0, start_index = 0) -> tuple[list[str], list[np.ndarray]]:
    """Grab either a specific number (if num_images is not None) or all the images from a
    directory you specify. 
    
    Note: to ensure we don't get predictions from already-computed images, if the image path already
    exists in the "database", we will skip over it.

    Args:
        image_dir (str): The directory to grab images from.
        image_paths (list): The already-predicted (from DB) image-paths (that we can skip over).
        num_images (int): The number of images you want to grab. If None, we grab all images.

    Raises:
        ValueError: If the number of images grabbed is equal to 0, then we throw an Error, as it's
        unlikely you want/expect to grab 0 (for gathering predictions for example).

    Returns:
        tuple(list[str], list[np.ndarray]): A tuple containing a list of image paths as strings, and a list of loaded images (as ndarrays).
    """
    loaded_images = []
    loaded_image_paths = []
    # curr_image_paths = grab_all_image_paths(image_dir)
    curr_index, num_loaded_images = 0, 0
    for path, dirs, files in os.walk(image_dir, topdown=True):
        for image_file_path in files:
            if curr_index < start_index:
                curr_index += 1
                continue
            curr_image_path = os.path.join(f'{path}/{image_file_path}')
            if curr_image_path not in image_paths:
                try:
                    curr_image = get_and_resize_image(curr_image_path, (299, 299, 3))
                    loaded_images.append(curr_image)
                    loaded_image_paths.append(curr_image_path)
                    num_loaded_images += 1
                except Exception as e:
                    # If we run into an error, just don't store and grab this image
                    pass
            if num_images and num_loaded_images > num_images:
                break
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

def resize_images(image_dir, new_image_dir):
    from PIL import Image
    import os, sys

    path = image_dir
    dirs = os.listdir( path )
    for item in dirs:
        img_path = f'{path}/{item}'

        if os.path.isfile(img_path):
            im = Image.open(img_path)
            f, e = os.path.splitext(img_path)
            imResize = im.resize((299,299), Image.ANTIALIAS)
            f = f'{f.split("/")[-1].split(".")[0]}'
            imResize.save(f'{new_image_dir}/{f+e}', 'JPEG', quality=90)