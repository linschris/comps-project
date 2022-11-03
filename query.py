import cProfile
import pstats
import numpy
from altered_xception import AlteredXception
from database import Database
from utils import get_youtube_urls, convert_to_embed_links

def query_image(query_img_path: str, callable_functions: list, k=5) -> list[list[str]]:
    """Queries image using different models and functions.

    Args:
        query_img_path (str): Image to be query
        callable_functions (list): A list of callable models (__call__) or functions
        k (int, optional): The number of top YouTube videos/URLs to compute. Defaults to 5.

    Returns:
        list[list[str]]: A list of the top YouTube URLs the model(s) and function(s) predicted.
    """
    model_outputs = []
    for function in callable_functions:
        if callable(function):
            image_scores = function(query_img_path)
            yt_urls = get_youtube_urls(image_scores)[0:k]
            convert_to_embed_links(yt_urls)
            model_outputs.append(yt_urls)
        else:
            print(f"{function.__name__} is not callable.")
    return model_outputs

def query_the_image(query_img_path, model, db, k=5):
    image_scores = query_image(query_img_path, db, k)
    yt_urls = get_youtube_urls(image_scores)[0:k]
    convert_to_embed_links(yt_urls)
    return yt_urls

