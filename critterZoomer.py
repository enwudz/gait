#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:25:57 2023

@author: iwoods

CODE to make video where tardigrade stays in place and background moves
    Turns are smoothed to move gradually to from old to new bearing
    
to make movie, run:
    python makeMovieFromImages.py searchterm fps outfile
    
WISHLIST
    add turn and stop labels to images
    see alpha code from demoTracking.py

    
"""

import cv2
import numpy as np
import sys
import gaitFunctions
import glob
import os
import scipy.signal

def main(movie_file, zoom_percent = 100):
    
    save_cropped_frames = False
    add_labels = True
    zoom_percent = 300
    
    # labeling stuff to adjust
    font = cv2.FONT_HERSHEY_DUPLEX # cv2.FONT_HERSHEY_SCRIPT_COMPLEX 
    text_size = 2
    turn_color = (0,0,0) # (155, 155, 0)
    stop_color = (0,0,0) #(15, 0, 100)
    time_x, time_y = [0.05, 0.1] # where should we put the time label?
    turn_x, turn_y = [0.05, 0.99] # where should we put the turn label?
    stop_x, stop_y = [0.5, 0.99] # where should we put the stop label?
    
    ''' check to see if rotated frames folder exists; if not, make a folder '''
    base_name = movie_file.split('.')[0]
    cropped_folder = base_name + '_rotacrop'
    flist = glob.glob(cropped_folder)
    if len(flist) == 1:
        print(' ... cropped frames already saved for ' + movie_file + '\n')
        # showFrames(cropped_folder)
        return cropped_folder
    
    # make a folder for the cropped frames if we want to save them
    if save_cropped_frames:
        os.mkdir(cropped_folder)

    ''' check for frame folder for this movie if none, create one and save frames '''
    # frame_folder = base_name + '_frames'
    # frame_folder = gaitFunctions.saveFrames(frame_folder, movie_file, add_labels)
    
    ''' get tracked path data and figure out bearing and cropping parameters '''
    # load tracked path data
    tracked_df, excel_filename = gaitFunctions.loadTrackedPath(movie_file)
    frametimes = tracked_df.times.values
    bearings = tracked_df.bearings.values
    turns = tracked_df.turns.values
    stops = tracked_df.stops.values

    if bearings[-1] == 0:
        bearings[-1] = bearings[-2]
    if bearings[0] == 0:
        bearings[0] = bearings[1]
    
    # smooth out bearing changes for abrupt turns and stops
    turn_ranges = gaitFunctions.one_runs(turns)
    for turn_range in turn_ranges:
        # get bearing before turn
        try:
            before_turn = bearings[turn_range[0]-1]
        except:
            before_turn = bearings[0]
            
        # get bearing after turn
        try: 
            after_turn = bearings[turn_range[1]]
        except:
            after_turn = bearings[-1]
            
        turn_length = turn_range[1] - turn_range[0]
        bearings[turn_range[0]:turn_range[1]] = np.linspace(before_turn, after_turn, turn_length)
   
    # smooth out the bearing changes, not so much movement
    pole = 3 # integer; lower = more smooth
    freq = 0.02 # float: lower = more smooth
    b, a = scipy.signal.butter(pole, freq)
    smoothed_bearings = scipy.signal.filtfilt(b,a,bearings)

    ## Quality control: compare bearings vs. smoothed bearings    
    # plt.plot(bearings,'r')
    # plt.plot(smoothed_bearings,'k')
    # plt.show()
    
    ''' Crop video window according to tardigrade size '''
    ## Get XY coordinates and tardigrade size
    smoothed_x = tracked_df.smoothed_x.values
    smoothed_y = tracked_df.smoothed_y.values
    mean_length = np.mean(tracked_df.lengths.values)
    
    ## set size of crop window
    crop_width_offset = int(mean_length * 0.4)
    crop_height_offset = int(mean_length * 0.8)
    
    ''' Get timing of turns and stops '''
    # these are arrays of frametimes when there are stops and turns
    stop_times = frametimes[np.where(stops==1)]
    turn_times = frametimes[np.where(turns==1)]
    
    # get alphas for labels of turns and stops
    label_buffer = 30 # in frames
    stop_alphas = gaitFunctions.labelTimes(frametimes, stop_times, label_buffer)
    turn_alphas = gaitFunctions.labelTimes(frametimes, turn_times, label_buffer)
    
    # where should we put the labels for frame time and stop and turn?
    vid = cv2.VideoCapture(movie_file)
    vid_width  = int(vid.get(4))
    vid_height = int(vid.get(3))
    time_stamp_position = (int(time_x * vid_width), int(time_y * vid_height) )
    turn_position =  (int(turn_x * vid_width), int(turn_y * vid_height) )
    stop_position = (int(stop_x * vid_width), int(stop_y * vid_height) )
    
    ''' go through movie and save frames '''

    print('.... saving frames!')

    base_name = movie_file.split('.')[0]
    
    i = 0
    while (vid.isOpened()):
        
        ret, im = vid.read()
        if ret == False: 
            break
        else:
            # found a frame
        
            # get data for this frame
            bearing = smoothed_bearings[i]
            rotate_angle = bearing # - 90
            x = smoothed_x[i]
            y = smoothed_y[i]
            
            # load frame image
            # im = cv2.imread(frame_file)
            
            # get center of image
            (h, w) = im.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            
            # get offset of centroid from center of image
            x_centroid_offset = x - cX
            y_centroid_offset = y - cY
            
            # pad image to twice max dimension
            padded = padImage(im, 100)
            # gaitFunctions.displayFrame(padded)
            
            # find new centroid after padding
            (padded_h, padded_w) = padded.shape[:2]
            (padded_cX, padded_cY) = (padded_w // 2, padded_h // 2)
            new_x = int(padded_cX + x_centroid_offset)
            new_y = int(padded_cY + y_centroid_offset)
            # cv2.circle(padded, (new_x, new_y), 10, (0,0,0), -1)
            
            # rotate padded image around centroid    
            M = cv2.getRotationMatrix2D((new_x, new_y), rotate_angle, 1.0)
            rotated = cv2.warpAffine(padded, M, (padded_w, padded_h))
            # gaitFunctions.displayFrame(rotated)    
            
            # crop around centroid
            ybot = new_y - crop_height_offset
            ytop = new_y + crop_height_offset
            xbot = new_x - crop_width_offset
            xtop = new_x + crop_width_offset
            
            cropped = rotated[ybot:ytop, xbot:xtop]        
            # gaitFunctions.displayFrame(cropped)  
            
            if zoom_percent == 100:
                frame = cropped
            else:
                frame = resizeImage(cropped, zoom_percent)
            
            if add_labels == True:
      
                # put the time variable on the video frame
                frame = cv2.putText(frame, str(frametimes[i]),
                                    time_stamp_position,
                                    font, text_size,
                                    (55, 55, 55),
                                    4, cv2.LINE_8)
                
                # add text for turns (fade in before and out after by text alpha)
                if turn_alphas[i] == 1:
                    cv2.putText(frame, 'Turn', turn_position, font, text_size, turn_color, 4, cv2.LINE_8)
                elif turn_alphas[i] > 0:
                    overlay = frame.copy()
                    cv2.putText(overlay, 'Turn', turn_position, font, text_size, turn_color, 4, cv2.LINE_8)
                    frame = cv2.addWeighted(overlay, turn_alphas[i], frame, 1 - turn_alphas[i], 0) 
                
                # add text for stops
                if stop_alphas[i] == 1:
                    cv2.putText(frame, 'Stop', stop_position, font, text_size, stop_color, 4, cv2.LINE_8)
                elif stop_alphas[i] > 0:
                    overlay = frame.copy()
                    cv2.putText(overlay, 'Stop', stop_position, font, text_size, stop_color, 4, cv2.LINE_8)
                    frame = cv2.addWeighted(overlay, stop_alphas[i], frame, 1 - stop_alphas[i], 0) 
            
            cv2.imshow('press (q) to quit', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            if save_cropped_frames:
                if frametimes[i] > 0: # cv2 sometimes(?) assigns the last frame of the movie to time 0            
                    file_name = base_name + '_rotacrop_' + str(int(frametimes[i]*1000)).zfill(6) + '.png'
                    cv2.imwrite(os.path.join(cropped_folder, file_name), frame)
          
        i += 1
     
    vid.release()
    return cropped_folder
    
    
def showFrames(frame_folder):
    frame_files = sorted(glob.glob(os.path.join(frame_folder, '*.png')))   
    for i, frame_file in enumerate(frame_files):
        # load frame image
        im = cv2.imread(frame_file)
        # show frames
        cv2.imshow('press (q) to quit', im)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    return
        
def resizeImage(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def cropAroundCenter(img, center, height_offset, width_offset):
    
    # if the cropping rectangle extends beyond the image, we need to
    # add pad of empty space pad around image
      
    pass    
    

def padImage(image, pad_percentage):
    old_image_height, old_image_width, channels = image.shape
    multiplier = 1 + pad_percentage / 100
       
    new_image_height = int(multiplier * old_image_height)
    new_image_width = int(multiplier * old_image_width)
    
    # print('padding image. Was ' + str(old_image_width) + 'x' + str(old_image_height))
    # print('Now: ' +  str(new_image_width) + 'x' + str(new_image_height))
    
    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    
    # make blank image (black)
    color = (0,0,0)
    padded = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
    
    padded[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = image
    
    return padded
    

if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])

    main(movie_file)
    
    
