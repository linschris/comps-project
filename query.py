import cProfile
import pstats
from faster_r_cnn import query_image
from database import Database


def query_img_by_similar_objects(query_img_path, db):
    return query_image(query_img_path, db)


def query_img_by_visual_similarity(query_img_path, db):
    pass


def get_youtube_urls(image_scores):
    youtube_urls = []
    for element in image_scores:
        image_path, score = element[0], element[1]
        yt_url = get_youtube_url(image_path)
        youtube_urls.append(yt_url)
    return youtube_urls


def get_youtube_url(image_path):
    curr_str = (image_path.split("/")[-1]).split("_frame")
    video_id = curr_str[0]
    if "." in video_id:
        return None
    if len(curr_str) == 2:  # We're looking at a frame
        time_str = get_youtube_time_str(curr_str[1][1:6])
        return f'https://www.youtube.com/watch?v={video_id}&t={time_str}'
    else:  # We're looking at a thumbnail
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


with cProfile.Profile() as pr:
    download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"
    db = Database(download_dir)
    image_scores = query_img_by_similar_objects(
        "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/biking3.jpg", db)
    print(get_youtube_urls(image_scores))

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename='needs_profiling.prof')
