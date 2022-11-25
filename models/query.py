from utils import get_youtube_urls

def query_image(query_img_path: str, models: list, k=5) -> dict:
    """Queries image using different models and functions.

    Args:
        query_img_path (str): Image to be query
        models (list): A list of models that can query the image by `query_image`
        k (int, optional): The number of top YouTube videos/URLs to compute. Defaults to 5.

    Returns:
        dict: A list of the top YouTube URLs each of the model(s) and function(s) predicted. 
        
        Note: frames from the same video will be grouped together.
    """
    model_outputs = {}
    for model in models:
        model_name = type(model).__name__
        query_image_method = getattr(model, "query_image")
        if callable(query_image_method): # If query_image method exists and is callable in model
            image_scores = model.query_image(query_img_path)
            yt_urls = get_youtube_urls(image_scores, k)
            model_outputs[model_name] = yt_urls
        else:
            print(f"{model_name} could not query the image.")
    return model_outputs