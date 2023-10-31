import skvideo.io # https://github.com/scikit-video/scikit-video do not use
                  # 1.1.1.0 and below since it's not compatible with numpy>=1.24
                  # Ctrl + F "Installing from github" on skvideo github page


import numpy as np
import cv2
import insightface
from roop.utilities import run_ffmpeg, move_temp
from roop.core import  decode_execution_providers
from face_watermark import add_water_mark, get_water_mark
import os
import shutil
from time import perf_counter



# THESE VALUES NEED TO BE CHANGED
target_path = './video.mp4'
source_path = './person.jpg'


output_path = './output.mp4'


def find_similar_face(frame, reference_face, similar_face_distance = 0.85):
    many_faces =FACE_ANALYSER.get(frame)
    if many_faces:
        for face in many_faces:
            if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                distance = np.sum(np.square(face.normed_embedding - reference_face.normed_embedding))
                if distance < similar_face_distance:
                    return face
    return None


def create_video(target_path, temp_directory_path, temp_output_path, output_video_quality = 35, output_video_encoder = 'libx264', temp_frame_format = 'png') -> bool:
    output_video_quality = (output_video_quality + 1) * 51 // 100
    print(output_video_quality)
    commands = ['-hwaccel', 'auto', '-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.' + temp_frame_format), '-c:v', output_video_encoder]
    if output_video_encoder in ['libx264', 'libx265', 'libvpx']:
        commands.extend(['-crf', str(output_video_quality)])
    if output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        commands.extend(['-cq', str(output_video_quality)])
    commands.extend(['-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])
    return run_ffmpeg(commands)

def restore_audio(target_path: str, output_path: str, temp_output_path) -> None:
    #temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(['-i', temp_output_path, '-i', target_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path])
    if not done:
        move_temp(target_path, output_path)


execution_providers = decode_execution_providers('cpu')

FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=execution_providers)
FACE_ANALYSER.prepare(ctx_id=0)

FACE_SWAPPER = insightface.model_zoo.get_model("./models/inswapper_128.onnx", providers=execution_providers)


t1_start = perf_counter() 

temp_directory_path = os.path.splitext(os.path.join("./temp/",(os.path.basename(target_path))))[0]
temp_output_path = os.path.join(temp_directory_path,'temp.mp4')


if not os.path.exists('./temp'):
    os.mkdir('./temp')


if not os.path.exists(temp_directory_path):
    os.mkdir(temp_directory_path)

metadata = skvideo.io.ffprobe(target_path)


height = int(metadata['video']['@height'])
width = int(metadata['video']['@width'])
fps = int(int(metadata['video']['@r_frame_rate'].split('/')[0]) / int(metadata['video']['@r_frame_rate'].split('/')[1]))
fps = min(fps, 30)

inputparameters = {}
outputparameters = {'-r': f'{fps}', "-pix_fmt":"bgr24"}
reader = skvideo.io.FFmpegReader(target_path,
                inputdict=inputparameters,
                outputdict=outputparameters)

source_face = FACE_ANALYSER.get(cv2.imread(source_path))[0]

reference_face = None
counter = 0
# iterate frames
for frame in reader.nextFrame():
    if counter == 0:
        water_mark = get_water_mark(frame.shape, "WATERMARK", 30, 45)

        # Only works if the reference face is in the first frame
        # It should be updated but since the original app does not do it
        # I did not change it to compare processing time
        reference_face = FACE_ANALYSER.get(frame)
        if reference_face:
            reference_face = reference_face[0]
        else:
            result = frame
    
    target_face = find_similar_face(frame, reference_face)
    if target_face:
        result = FACE_SWAPPER.get(frame, target_face, source_face, paste_back=True)
    else:
        result = frame
    counter += 1
    result = add_water_mark(result, water_mark, 0.1)
    image_path = os.path.join(temp_directory_path, f'{counter:04}.png')
    cv2.imwrite(image_path, result)


create_video(target_path, os.path.abspath(temp_directory_path), temp_output_path)
restore_audio(target_path, output_path, temp_output_path)

 
shutil.rmtree(temp_directory_path)
if os.path.exists('./temp') and not os.listdir('./temp'):
    os.rmdir('./temp')
t1_stop = perf_counter()
print("Elapsed time:", t1_stop - t1_start) 


