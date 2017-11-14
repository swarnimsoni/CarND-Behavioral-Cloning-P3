from moviepy.editor import ImageSequenceClip
from moviepy.editor import VideoFileClip

import argparse
import cv2

color_space = ''
def convertImage(image):
    global color_space
    feature_image=[]
    if color_space == 'HSV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return feature_image #cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

def main():
    global color_space
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'inputVideoFile',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
#    parser.add_argument(
#        'outputVideoFile',
#        type=str,
#        default='',
#        help='Path to image folder. The video will be created from these images.'
#    )
#    parser.add_argument(
#        '--fps',
#        type=int,
#        default=60,
#        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()



    color_space_array = ['HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']

    for i_colorSpace in color_space_array:

#        print("Creating video {}, FPS={}".format(video_file, args.fps))
        color_space = i_colorSpace

        clip = VideoFileClip(args.inputVideoFile)
        processedVideo = clip.fl_image(convertImage)
        processedVideoFileName = i_colorSpace + args.inputVideoFile
        processedVideo.write_videofile(processedVideoFileName, audio=False)

if __name__ == '__main__':
    main()
