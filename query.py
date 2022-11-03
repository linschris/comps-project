import cProfile
import pstats
import numpy
from altered_xception import AlteredXception
from database import Database
from utils import get_youtube_urls

def query_image(query_img_path: str, callable_functions: list, k=5) -> list[list[str]]:
    """Queries image using different models and functions.

    Args:
        query_img_path (str): Image to be query
        callable_functions (list): A list of callable models (__call__) or functions
        k (int, optional): The number of top YouTube videos/URLs to compute. Defaults to 5.

    Returns:
        list[list[str]]: A list of the top YouTube URLs the model(s) and function(s) predicted.
    """
    model_outputs = {}
    for function in callable_functions:
        if callable(function):
            image_scores = function(query_img_path)
            yt_urls = get_youtube_urls(image_scores, k)
            model_outputs[type(function).__name__] = yt_urls
        else:
            print(f"{type(function).__name__} is not callable.")
    return model_outputs