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
    
"""

import cv2
import numpy as np
import sys
import gaitFunctions
import glob
import os
# import scipy.signal

def main(cropped_folder, movie_file, zoom_percent = 100, save_cropped_frames = False, starttimes = [], endtimes = []):
    
    ''' ----> Adjust these parameters for labels '''
    font = cv2.FONT_HERSHEY_DUPLEX # cv2.FONT_HERSHEY_SCRIPT_COMPLEX 
    add_labels = True # True to add time stamp and turn / stop labels
    text_size = 1
    turn_color = (155,155,0) # (155, 155, 0) # all zeros for nothing
    stop_color = (15,0,100) # (15, 0, 100) # all zeros for nothing
    time_x, time_y = [0.05, 0.1] # where should we put the time label?
    turn_x, turn_y = [0.05, 0.95] # where should we put the turn label?
    stop_x, stop_y = [0.5, 0.95] # where should we put the stop label?
    
    # report selections
    print('\nMovie is ' + movie_file)
    print('Zoom is ' + str(zoom_percent) + ' percent')
    
    ''' check to see if rotated frames folder exists; if not, make a folder '''
    flist = glob.glob(cropped_folder)
    
    if len(flist) == 1:
        print('... ' + cropped_folder + ' ... already exists \n')
        
        # check to see if images in this folder
        searchterm = os.path.join(cropped_folder, '*.png')
        if len(glob.glob(searchterm)) > 0:
            print('... found rotated and cropped images in ' + cropped_folder)
            return cropped_folder
        else:
            print('... no images in ' + cropped_folder + ' ... will save them now.')
            save_cropped_frames = True
    
    # make a folder for the cropped frames if we want to save them
    if save_cropped_frames is False:
        selection = input('\nShould we save rotated and cropped frames? (y) or (n): ')
        if selection == 'y':
            save_cropped_frames = True
            
    if save_cropped_frames:        
        os.mkdir(cropped_folder)
    
    ''' get tracked path data and figure out bearing and cropping parameters '''
    # load tracked path data
    tracked_df, excel_filename = gaitFunctions.loadTrackedPath(movie_file)
    frametimes = tracked_df.times.values
    bearings = tracked_df.filtered_bearings.values # was just bearings
    # delta_bearings = tracked_df.bearing_changes.values
    turns = tracked_df.turns.values
    stops = tracked_df.stops.values

    # pad boundaries of bearings and delta_bearings
    bearings = padBoundaries(bearings)
    
    # determine if there is a consistent direction of travel
    if np.var(bearings) < 45:
        
        # see if this is a tardigrade ... if so, direction does not matter
        identity_info = gaitFunctions.loadIdentityInfo(movie_file)
        try:
            species = [identity_info['species']]
        except:
            species = 'tardigrade'
        
        # if not a tardigrade, it might be going fast in a particular direction
        if species == 'tardigrade':
            direction = 'up'
            live_rotate = True
        else:
            wiggle_room = 10
            mean_bearings = np.mean(bearings)
            if 270-wiggle_room < mean_bearings < 270+wiggle_room:
                direction = 'left'
                live_rotate = False
            elif 90-wiggle_room < mean_bearings < 90+wiggle_room:
                direction = 'right'
                live_rotate = False
            elif 180-wiggle_room < mean_bearings < 180+wiggle_room:
                direction = 'down'
                live_rotate = False
            elif 0-wiggle_room < mean_bearings < 0+wiggle_room:
                direction = 'up'
                live_rotate = False
            else:
                direction = 'up'
                live_rotate = True
                
    print('\nWill show direction of traval as ' + direction)
    if live_rotate:
        print('... direction of travel varies; will rotate every frame')
    else:
        print('... direction of travel is constant; will not rotate frames')
    
    ## smooth out the bearings or bearing changes, not so much movement
    # pole = 10 # integer; lower = more smooth ... but 'freq' has more effect?
    # freq = 0.04 # float: lower = more smooth ... has more effect than 'pole'?
    # b, a = scipy.signal.butter(pole, freq)
    # smoothed_deltabearings = scipy.signal.filtfilt(b,a,delta_bearings)
    # smoothed_bearings = scipy.signal.filtfilt(b,a,bearings)

    # Quality control for smoothing: compare bearing changes vs. smoothed bearing changes   
    # import matplotlib.pyplot as plt
    # plt.plot(bearings,'r')
    # plt.plot(smoothed_bearings,'k')
    # plt.show()
    # exit()
    
    ''' Crop video window according to critter size ... '''
    ## Get XY coordinates and critter size
    ## note that length is the body dimension in the direction of travel  
    smoothed_x = tracked_df.smoothed_x.values
    smoothed_y = tracked_df.smoothed_y.values
    mean_length = np.mean(tracked_df.lengths.values)
    mean_width = np.mean(tracked_df.widths.values)
    if mean_length == 0:
        print('Need to measure critter dimensions!')
        import manualCritterMeasurement
        critter_length, critter_width = manualCritterMeasurement.main(movie_file)
    else:
        critter_length, critter_width = mean_length, mean_width
    
    ### set size of crop window based on size of critter (or set manually below)
    print('Cropping frames based on length of ' + str(np.round(critter_length,1)) + ' pixels in direction of travel')
    length_multiplier = 2.5
    width_multiplier = 0.8
    
    if direction in ['up','down']:
        # image height scaled to body LENGTH
        # image width scaled to body WIDTH
        crop_height_offset = int(critter_length * length_multiplier)
        crop_width_offset = int(critter_width * width_multiplier) 
        
    else:
        # image height scaled to body WIDTH
        # image width scaled to body LENGTH
        crop_height_offset = int(critter_width * width_multiplier)
        crop_width_offset = int(critter_length * length_multiplier) 
    
    ### OR MANUALLY set cropping size  . . . 
    # crop_width_offset = 100 # set to half of desired width # 100 for representative movies
    # crop_height_offset = 180 # set to half of desired height # 180 for representative movies
    
    print('Cropped width offset:  ', crop_width_offset, 'pixels')
    print('Cropped height offset: ', crop_height_offset, 'pixels')
    
    # get dimensions of cropped video
    vid_width = 2 * crop_width_offset
    vid_height = 2 * crop_height_offset
    
    print('Cropped Height x Width = ', vid_height, 'x', vid_width)

    ''' Get timing of turns and stops '''
    # these are arrays of frametimes when there are stops and turns
    stop_times = frametimes[np.where(stops==1)]
    turn_times = frametimes[np.where(turns==1)]
    
    # get alphas for labels of turns and stops
    label_buffer = 30 # in frames
    stop_alphas = gaitFunctions.labelTimes(frametimes, stop_times, label_buffer)
    turn_alphas = gaitFunctions.labelTimes(frametimes, turn_times, label_buffer)
    # print(turn_alphas) # testing
    
    # where should we put the labels for frame time and stop and turn?
    vid = cv2.VideoCapture(movie_file)
    # vid_width  = int(vid.get(4))
    # vid_height = int(vid.get(3))
    time_stamp_position = (int(time_x * vid_width), int(time_y * vid_height) )
    turn_position =  (int(turn_x * vid_width), int(turn_y * vid_height) )
    stop_position = (int(stop_x * vid_width), int(stop_y * vid_height) )
    
    ''' go through movie and save frames '''
    if save_cropped_frames:
        print('.... saving frames!')

    base_name = movie_file.split('.')[0]
    
    # rotate image based on desired direction of travel
    flipToRight = False
    
    if direction == 'down':
        initial_direction = 180
    elif direction == 'left':
        initial_direction = -90 
        flipToRight = True
        print('We will flip this to the right!')
    elif direction == 'right':
        initial_direction = 90
    else:
        initial_direction = bearings[0]
    print('Initial heading is ', initial_direction) 
    
    rotate_angle = initial_direction
    
    i = 0
    while (vid.isOpened()):
        
        ret, im = vid.read()
        if ret == False: 
            break
        else:
            # found a frame
            
            # get data for this frame
            # rotate_angle += smoothed_deltabearings[i]
            rotate_angle = bearings[i]
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
            if live_rotate: # variable direction, need to rotate every frame
                M = cv2.getRotationMatrix2D((new_x, new_y), rotate_angle, 1.0)
                rotated = cv2.warpAffine(padded, M, (padded_w, padded_h))
            else: # constant direction
                rotated = padded
            # gaitFunctions.displayFrame(rotated)    
            
            # crop around centroid
            ybot = new_y - crop_height_offset
            ytop = new_y + crop_height_offset
            xbot = new_x - crop_width_offset
            xtop = new_x + crop_width_offset
            
            cropped = rotated[ybot:ytop, xbot:xtop]        
            # gaitFunctions.displayFrame(cropped)  
            
            if flipToRight:
                cropped = cv2.flip(cropped, 1)
            
            if zoom_percent == 100:
                frame = cropped
            else:
                frame = resizeImage(cropped, zoom_percent)
            
            if add_labels == True:
      
                # put the time variable on the video frame
                frame = cv2.putText(frame, str(frametimes[i]),
                                    time_stamp_position,
                                    font, text_size,
                                    (255, 255, 255), # (55, 55, 55)
                                    4, cv2.LINE_8)
                
                # add text for turns (fade in before and out after by text alpha)
                if turn_alphas[i] == 1:
                    # print('adding turn') # testing
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
                
                save_this_frame = False
                
                # decide whether to save this particular frame
                # cv2 sometimes(?) assigns the last frame of the movie to time 0  
                if len(starttimes) == 0 and frametimes[i] > 0:
                    save_this_frame = True
                elif len(starttimes) > 0 and len(endtimes) > 0:
                    # see if this frame is within a bout
                    if gaitFunctions.numberWithinRanges(frametimes[i], starttimes, endtimes):
                        save_this_frame = True
                
                if save_this_frame:           
                    file_name = base_name + '_rotacrop_' + str(int(frametimes[i]*1000)).zfill(6) + '.png'
                    cv2.imwrite(os.path.join(cropped_folder, file_name), frame)
              
        i += 1
     
    vid.release()
    
    selection = input('\nSave video from rotated and cropped images? (y) or (n): ').rstrip()
    if selection == 'y':
        import makeMovieFromImages
        searchterm = os.path.join(cropped_folder, '*rotacrop*')
        savefile = os.path.join(cropped_folder, base_name + '_rotacrop.mp4')
        makeMovieFromImages.main(searchterm, 30, savefile)
    
        selection = input('\nRemove frame images in ' + cropped_folder + '? (y) or (n): ').rstrip()
        if selection == 'y':
            searchterm = os.path.join(cropped_folder, '*.png')
            frame_files = glob.glob(searchterm)
            for frame in frame_files:
                os.remove(frame)
    
    return cropped_folder
 
def padBoundaries(a):
    # replace leading and trailing zeros with inner values
    if a[0] == 0:
        a[0] = a[1]
    if a[-1] == 0:
        a[-1] = a[-2]
    return a
    
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
 
def getCroppedFolder(movie_file):
    base_name = movie_file.split('.')[0]
    cropped_folder = base_name + '_rotacrop'
    return cropped_folder

if __name__== "__main__":
    
    zoom_percent = 100
    
    if len(sys.argv) == 1:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])     
    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    if len(sys.argv) > 2:
        zoom_percent = int(sys.argv[2])

    cropped_folder = getCroppedFolder(movie_file)
    main(cropped_folder, movie_file, zoom_percent)
    
    
    
