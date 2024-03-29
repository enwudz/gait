#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 21:02:29 2022

@author: iwoods

ffmpeg:

   ffmpeg -f image2 -r 30 -pattern_type glob -i '*_frames_*.png' -pix_fmt yuv420p -crf 20 demo_outline_centroid_cool.mp4
   https://video.stackexchange.com/questions/18547/simple-video-editing-software-that-can-handlethis/18549#18549

Wish list:
1. if cannot find object in first frame, cX and cY will be undefined
and program will exit with an error
so need to figure out what to do if this is the case


"""

import cv2
import numpy as np
from matplotlib import colormaps as cm
import glob
import sys
from scipy import stats
import gaitFunctions
import pandas as pd
import os

def main(movie_file, difference_threshold = 25, showTracking = True):
    
    # get or make excel file for this clip
    excel_file_exists, excel_filename = gaitFunctions.check_for_excel(movie_file)
    if excel_file_exists == False:
        import initializeClip
        initializeClip.main(movie_file)
        
    # check if the tracking data is already there ... ask if want to run again
    tracking_data = pd.read_excel(excel_filename, sheet_name='pathtracking')
    if 'xcoords' in tracking_data.columns:
        print('... looks like there is already tracking data for this clip!')
        skip = continue_tracking()
        if skip == 'skip':
            return
    
    # get first and last frames
    first_frame, last_frame = gaitFunctions.getFirstLastFrames(movie_file)
        
    # make background image using N random frames of video
    # (or load a background image that is already made)
    background_image = backgroundFromRandomFrames(movie_file, 100)

    # run through video and compare each frame with background
    centroid_coordinates, areas, lengths, widths, problem_frames = findCritter(movie_file, background_image, difference_threshold, showTracking) # typical threshold is 25, scale of 1 to 255
    
    # save the centroid_coordinates and areas to the excel file
    times = [x[0] for x in centroid_coordinates]
    xcoords = [x[1] for x in centroid_coordinates]
    ycoords = [x[2] for x in centroid_coordinates]
    
    tracking_uncertainty_key = 'tracking_uncertainty_' + str(difference_threshold) 
    d = {'times':times, 'xcoords':xcoords, 'ycoords':ycoords, 
         'areas':areas, 'lengths':lengths, 'widths':widths, tracking_uncertainty_key:problem_frames}
    
    # report on tracking confidence
    gaitFunctions.getTrackingConfidence(problem_frames, difference_threshold, True)

    # save data
    print('Saving tracking data for ' + movie_file)
    df = pd.DataFrame(d)
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        df.to_excel(writer, index=False, sheet_name='pathtracking')

    return df

def continue_tracking():
    selection = input('\n ... (r)edo the tracking, (s)kip for now, or (q)uit entirely?  ')
    if selection == 's':
        return 'skip'
    elif selection != 'r':
        exit('')
    return ''

def findCritter(video_file, background, pixThreshold, showTracking):

    # go through video
    print('... starting video ' + video_file)
    vid = cv2.VideoCapture(video_file)
    fps = vid.get(5)

    # print out video info
    printVidInfo(vid, True)

    print('... using pixel threshold ' + str(pixThreshold))

    frame_number = 0
    frames_in_video = getFrameCount(video_file)
    dot_colors = makeColorList('plasma', frames_in_video) # cool

    centroid_coordinates = [] # container for (x,y) coordinates of centroid of target object at each frame
    areas = [] # container for calculated areas of target object at each frame
    lengths = [] # container for calculated lengths of target object at each frame
    widths = []  # container for calculated widths of target object at each frame
    
    stored_target = ''
    problem_frames = [] # list to keep track of frames with trouble tracking

    while vid.isOpened():
        
        ret, frame = vid.read()
        frame_number += 1
        problem_frame = False
        
        # frameTime = int(vid.get(cv2.CAP_PROP_POS_MSEC)) # this returns zeros at end of video
        frameTime = round(float(frame_number)/fps,4)

        if ret != True:  # no frame!
            print('... video end!')
            break

        # save a copy of the original frame for later
        # saved_frame = frame.copy()
        
        # find difference between frame and background, as thresholded binary image
        binary_frame = getBinaryFrame(frame, background, pixThreshold)

        # find contours on the thresholded binary image
        contours, hierarchy = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # if more than one contour, make a decision about which contour is the target object
        # decision can be based on area of target ... or maybe last known position?
        if len(contours) > 1:
            print('frame ' + str(frame_number) + ' has ' + str(len(contours)) + ' detected objects!')
            problem_frame = True
            if len(areas) == 0:
                target_area = 14000 # just a guess
                midwidth,midheight = np.round(np.shape(frame)[:2])
                current_loc = (midwidth,midheight)
            else:
                target_area = np.mean(areas)
                current_x = centroid_coordinates[-1][1]
                current_y = centroid_coordinates[-1][2]
                current_loc = np.array([current_x, current_y])
            target = getTargetObject(contours, stored_target, current_loc, target_area)
        else:
            target = contours[0]
    
        stored_target = target
    
        # calculate moment of target object
        M = cv2.moments(target)

        if M["m00"] > 0:
            
            # get centroid of target object
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # get area of target object
            target_area = cv2.contourArea(target)
            
            # get length and width of target object
            center_points, (w,h), rotation_angle = cv2.minAreaRect(target)
            if h >= w:
                target_length = h
                target_width = w
            else:
                target_length = w
                target_width = h
            
        else:
            print('skipping this frame, cannot find target object')
            # to keep #centroids == #frames ...
            # append last centroid found

        # if cannot find object in first frame, cX and cY will be undefined
        # and program will exit with an error
        # so need to figure out what to do if this is the case

        # store coordinates
        centroid_coordinates.append((frameTime,cX,cY))
        areas.append(target_area)
        lengths.append(target_length)
        widths.append(target_width)
        problem_frames.append(problem_frame)

        if showTracking:
            # ==> SHOW CONTOURS: on the original frame
            cv2.drawContours(frame, contours, -1, (0,255,0), 5)
    
            #  show (color coded) centroids on frame
            cv2.circle(frame, (cX, cY), 5, dot_colors[frame_number-1], -1)
            # ==> OR show ALL centroids so far on the frame
            frame  = addCoordinatesToFrame(frame, centroid_coordinates, dot_colors[:len(centroid_coordinates)])
    
            # ==> SHOW TIME STAMPS: show (color coded) time stamps on frame
            # put the time variable on the video frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, str(frameTime).ljust(5,'0'),
                                (100, 100), # position
                                font, 1, # font, size
                                dot_colors[frame_number-1], # color
                                4, cv2.LINE_8)
            
            # ==> SHOW THE MOVIE (with centroids, times, or whatever is added)
            cv2.imshow('press (q) to quit', frame) # frame or binary_frame
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # ==> SAVE FRAME TO FILE
        # fstem = video_file.split('.')[0]
        # saveFrameToFile(fstem, frameTime, frame) # frame or binary_frame

   
    # shut down the video capture object
    vid.release()
    cv2.destroyAllWindows()
    
    return centroid_coordinates, areas, lengths, widths, problem_frames


def addCoordinatesToFrame(frame, coordinates, colors):
    '''

    Parameters
    ----------
    frame : open CV image
    coordinates : list of tuples
        a list of tuples of coordinates within frame
    colors : list of colors
        a list of tuples of cv2 formatted colors ... can be longer than coordinates list

    Returns
    -------
    frame with a dot positioned at each coordinate

    '''
    for i, coord in enumerate(coordinates):
        cv2.circle(frame, (coord[1], coord[2]), 5, colors[i], -1)

    return frame

def makeColorList(cmap_name, N):
     cmap = cm[cmap_name]
     cmap = cmap(np.linspace(0,1,N))[:,0:3]
     cmap = np.fliplr(cmap)

     # format for cv2 = 255 is max pixel intensity, colors are BGR
     cmap = cmap * 255 # for opencv colors
     # convert RGB to BGR ... apparently no need!
     # cmap = [[color[0], color[1], color[2]] for i, color in enumerate(cmap)]

     return [tuple(i) for i in cmap]

def getTargetObject(contours, stored_contour, current_loc, targetArea=5000):
    '''
    Parameters
    ----------
    contours : list of contours from cv2.findContours
        object is white, background is black
    current_loc : tuple of coordinates of last known position of object
    targetArea : integer
        expected or calculated area (in pixels) of target object.

    Returns
    -------
    the contour from within contours that is closest in position (and area)

    '''

    # find the contour that is closest to the last known position
    moments = [cv2.moments(cnt) for cnt in contours]
    try:
        xcoords = [int(M["m10"] / M["m00"]) for M in moments]
        ycoords = [int(M["m01"] / M["m00"]) for M in moments]
    except:
        print(' ... trouble finding object')
        if len(stored_contour) > 0:
            return stored_contour
        else:
            return contours[0]

    coords = [np.array([x,ycoords[i]]) for i,x in enumerate(xcoords)]
    distances = [np.linalg.norm(current_loc - coord) for coord in coords]
    closest_in_distance = np.argmin(distances)

    # find the contour with the closest area
    areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    closest_in_area = np.argmin(np.abs(areas - targetArea))

    if closest_in_distance != closest_in_area:
        
        # which is the better guess? distance or area?
        # if distance > the length of the tardigrade, then choose AREA
        length_estimate = 1.2 * np.sqrt(targetArea)
        min_distance = min(distances)
        if min_distance > length_estimate:
            best_guess = contours[closest_in_area]
            print(' ... target uncertain, choosing the one that is closest to expected size')
        else:
            best_guess = contours[closest_in_distance]
            print(' ... target uncertain, choosing the one that is closest to expected location')
        
    else:
        print(' ... choosing the target that is closest in position and size')
        best_guess = contours[closest_in_distance]
        
    return best_guess

def getBinaryFrame(frame, background, pixThreshold):

    # convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # do NOT blur this frame. Blurring will create differences with the background.
    # but here's code to blur, just to keep it handy
    # gaussian kernel size
    # needs to be positive, and odd. Higher values = more blurry
    #kernelSize = (51,51)
    #blurred_frame =  cv2.GaussianBlur(gray_frame,kernelSize,0)
    #blurred_frame =  cv2.medianBlur(frame,kernelSize[0])

    # find difference from background
    diff = cv2.absdiff(background, gray_frame)

    # convert difference to binary image
    _,diffbw = cv2.threshold(diff, pixThreshold, 255,cv2.THRESH_BINARY)
    #_,diffbw = cv2.threshold(diff,pixThreshold,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # blur the binary image ... this can get rid of 'holes' within contours
    kernelSize = (21,21) # higher kernel = more blurry, slower
    blurred_diff = cv2.medianBlur(diffbw, kernelSize[0]) # medianBlur slow ... but seems to do better
    #blurred_diff =  cv2.GaussianBlur(diffbw,kernelSize,0) # gaussian faster than median

    # threshold again to omit pixels of intensity < 255 (from the blur)
    # ... or leave for later as part of contour picking?
    _,final_diff = cv2.threshold(blurred_diff, 120, 255,cv2.THRESH_BINARY)

    return final_diff

def saveFrameToFile(file_stem, frameTime, frame):
    # to make a movie from frames
    # conda install ffmpeg
    # ffmpeg -f image2 -r 10 -s 1080x1920 -pattern_type glob -i '*.png' -vcodec mpeg4 movie.mp4
    # -r is framerate of movie
    timestamp = str(int(frameTime*1000)).zfill(6)

    file_name = file_stem + '_frames_' + timestamp + '.png'
    cv2.imwrite(file_name, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def loadBackgroundImage(imagepath):
    print('... loading background image ... ')
    background_image = cv2.imread(imagepath)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    return background_image

def checkForBackgroundImage(movie_file):
    
    # check to see if there is already an image
    background_imname = movie_file.split('.')[0] + '_background.png'
    background_exists = False
    imagepath = ''
    
    if len(glob.glob(background_imname)) > 0:
        background_exists = True
        imagepath = background_imname
    
    elif len(glob.glob(os.path.join('background_images', background_imname))) > 0:
        background_exists = True
        imagepath = os.path.join('background_images', background_imname)
    
    if background_exists:
        background_image = loadBackgroundImage(imagepath)        
    else:
        background_image = np.array([])
    
    return background_imname, background_image, background_exists

def backgroundFromRandomFrames(movie_file, num_background_frames):
    ''' Check if there is already a background image for this movie file
    if so, load it and return it
    if not, make one and return it'''

    # maybe try https://hostadvice.com/how-to/how-to-do-background-removal-in-a-video-using-opencv/ instead?

    background_imname, background_image, background_exists = checkForBackgroundImage(movie_file)
    
    if background_exists == False:

        # if no background image already, make one!
        
        # Select n frames at random from the video
        frames_in_video = getFrameCount(movie_file)
        
        num_background_frames = min(num_background_frames, frames_in_video)
        
        background_frames = sorted(np.random.choice(frames_in_video,
                                       num_background_frames,
                                       replace = False))

        # make an empty matrix to store stack of video frames
        img = getFirstFrame(movie_file) # get first frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_height, img_width = np.shape(gray)
        video_stack = np.zeros([img_height, img_width, len(background_frames)])

        frame_counter = 0
        image_index = 0

        print("... no background image yet! Making one now ... ")

        # go through video
        vid = cv2.VideoCapture(movie_file)
        while (vid.isOpened()):
            
            ret, frame = vid.read()
            
            # occasionally an empty frame will be encountered before the expected end of the video
            if (ret != True):  # no frame!
                print('... empty image at frame ' + str(frame_counter+1) + '!')
                print('... expected ' + str(frames_in_video) + ' frames in ' + movie_file)
                video_stack = video_stack[:,:,:image_index]
                break

            #print('Looking at frame ' + str(frame_counter) + ' of ' + str(frames_in_video))
            if frame_counter in background_frames:
                
                if (image_index + 1) % 50 == 0:
                    print('     getting frame ' + str(image_index+1) + ' of ' + str(num_background_frames) )
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                video_stack[:,:,image_index] = gray_frame
                image_index += 1
            frame_counter += 1

        #print('... releasing video ...')
        vid.release()
        
        # get mode of image stack for each pixel
        print("... calculating mode for background image (takes awhile) ...")
        background_image = stats.mode(video_stack, axis=2, keepdims=True)[0][:,:] # this is SLOW!
        
        # consider blurring the background image a bit?

        print('... saving background image as ' + background_imname)
        
        if len(glob.glob('background_images')) == 0:
            os.mkdir('background_images')
        background_impath = os.path.join('background_images', background_imname)
        
        cv2.imwrite(background_impath, background_image)
        print('... finished making background image!')
        
        # klugey to load it again when we already have it!
        background_image = loadBackgroundImage(background_impath)

    return background_image

# def getBackgroundFrames(num_frames, num_background_frames):
#     num_background_frames = min(num_background_frames, num_frames)

#     return sorted(np.random.choice(num_frames,
#                                    num_background_frames,
#                                    replace = False)), num_background_frames

def getImageSize(img):
    """get the # rows x # cols for an image"""
    num_rows, num_cols = np.shape(img)[:2]
    return num_rows, num_cols

def getFrameCount(movie_file):
    """get the number of frames in a movie file"""
    excel_ok = True
    excel_filename = movie_file.split('.')[0] + 'xlsx'
    
    try:
        info_df = pd.read_excel(excel_filename, sheet_name='identity', index_col=None)
    except:
        excel_ok = False
        
    if excel_ok:
        info = dict(zip(info_df['Parameter'].values, info_df['Value'].values))
    else:
        import initializeClip
        info = initializeClip.main(movie_file, False) # False is whether or not to print
    
    return info['#frames']

def getFirstFrame(videofile):
    """get the first frame from a movie, and return it as an image"""
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        return image
    else:
        print('cannot get an image from ' + videofile)
        return None


def displayImage(img):
    """Show an image, press any key to quit"""
    cv2.imshow('press any key to exit', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def printVidInfo(vid, printIt = True):
    vid_width  = int(vid.get(3))
    vid_height = int(vid.get(4))
    vid_fps = int(np.round(vid.get(5)))
    vid_frames = int(vid.get(7))
    if printIt is True:
        print('  Video height: ' + str(vid_height))
        print('  Video width:  ' + str(vid_width))
        print('  Video fps:    ' + str(vid_fps))
        print('  Video frames: ' + str(vid_frames))
    return (vid_width, vid_height, vid_fps, vid_frames)

if __name__== "__main__":

    showtracking = True
    
    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])
       
    print('Movie is ' + movie_file)

    try:
        difference_threshold = int(sys.argv[2])
    except:
        difference_threshold = 12
        
    try:
        show_track = sys.argv[3].lower()
        if show_track == 'false':
            showtracking = False
    except:
        showtracking = True

    main(movie_file, difference_threshold, showtracking)
