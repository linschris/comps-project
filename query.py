import cProfile
import pstats
import numpy
from altered_xception import AlteredXception
from database import Database


# def query_img_by_similar_objects(query_img_path, db):
    
#     return query_image(query_img_path, db)


def query_img_by_visual_similarity(query_img_path, model):
    '''Bad: purely to see how well this does.'''
    query_fv = model.grab_image_feature_vector(query_img_path)
    distances = []
    for image_path, index in model.db.prediction_image_paths.items():
        curr_index = model.db.prediction_image_paths[image_path]
        curr_fv = model.db.predictions[curr_index]
        dist = numpy.linalg.norm(query_fv - curr_fv)
        distances.append([image_path, dist])    
    return sorted(distances, key=lambda x: x[1])


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

def query_the_image(query_img_path, model, db, k=5):
    image_scores = query_img_by_visual_similarity(query_img_path, model)
    yt_urls = get_youtube_urls(image_scores)[0:k]
    convert_to_embed_links(yt_urls)
    return yt_urls

# with cProfile.Profile() as pr:
#     download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data/predictions"
#     db = Database(download_dir)
#     model = AlteredXception(db)
#     image_scores = query_img_by_visual_similarity(
#         "/Users/lchris/Desktop/Taka_Shiba.jpeg", model)
#     print(get_youtube_urls(image_scores))

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.dump_stats(filename='needs_profiling.prof')
