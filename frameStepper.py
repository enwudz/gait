#!/usr/bin/python
import cv2
import sys
import os
import glob
import gaitFunctions
import pandas as pd
from time import sleep

# aside: want to make a movie from a bunch of frames?
# brew install ffmpeg
# ffmpeg below works Oct 2022
# ffmpeg -f image2 -r 30 -pattern_type glob -i '*_frames_*.png' -pix_fmt yuv420p -crf 20 demo_movie.mp4
# -r is framerate of movie

'''
WISH LIST

1. allow to change mind . . . e.g. hit 'x' if want to clear current entry
'''

def main(movie_file, resize=100):

    # in terminal, navigate to the directory that has your movies
    
    # load excel file for this clip
    excel_file_exists, excel_filename = gaitFunctions.check_for_excel(movie_file)
    if excel_file_exists:
        # df = pd.read_excel(excel_filename, sheet_name='identity', index_col=None)
        # info = dict(zip(df['Parameter'].values, df['Value'].values))
        
        # check if there is any step data already; load if so
        foot_data_df = pd.read_excel(excel_filename, sheet_name='steptracking', index_col=None)
        
        if len(foot_data_df) > 1:
            # load foot_data dictionary
            foot_data = dict(zip(foot_data_df['leg_state'].values,foot_data_df['times'].values))
            # convert foot_data from string to list, to match data collection below
            for leg_state in foot_data.keys():
                if len(foot_data[leg_state]) > 0:
                    foot_data[leg_state] = foot_data[leg_state].split(' ')
        else:
            # make a foot_data dictionary
            foot_data = {}
    
    else:
        import initializeClip
        initializeClip.main(movie_file)
        foot_data = {}

    # look for frame folder for this movie
    # if none there, create one and save frames
    frame_folder = movie_file.split('.')[0] + '_frames'
    frame_folder = saveFrames(frame_folder, movie_file)

    feet_done = getFeetDone(foot_data)
    feet_to_do = getFeetToDo(feet_done)
    finished_tracking = False
    if len(feet_to_do) > 0:
        print('Still need to track steps for: ' + ', '.join(feet_to_do))
    else:
        print('Already finished all legs for this movie!')
        finished_tracking = True
        
    for foot in feet_to_do:
        
        print('\nNext foot to do is ' + foot + ' ...')
        selection = input('     (t)rack or (q)uit ? ')
        
        if selection != 't':
            print('ok, we are done for now!')
            break
        
        else:

            print('... record data for ' + foot + '\n')
        
            # step through frames and label things
            # can enter a number for resize to scale video
            data = stepThroughFrames(frame_folder, foot, resize) 
        
            # print out foot down and foot up data for this foot
            foot_step_times = showFootDownUp(foot, data)
            print(foot_step_times)
        
            # add foot data to the foot dictionary
            # data[0] is down times, data[1] is up times
            foot_data[foot+'_down'] = data[0]
            foot_data[foot+'_up'] = data[1]
    
    # all done, save data!
    saveData(excel_filename, foot_data)

    # R4 is the last foot - could ask to do next steps after finishing
    if finished_tracking == True:
        print('\nAlready tracked steps for ' + movie_file)
        import analyzeSteps
        analyzeSteps.main(movie_file)
        
    elif foot == 'R4':
        selection = input('\n... all done with feet, (r)un analyzeSteps.py? ')
        if selection == 'y' or selection == 'r':
            import analyzeSteps
            analyzeSteps.main(movie_file)
        gaitFunctions.removeFramesFolder(movie_file)
        gaitFunctions.cleanUpTrash(movie_file)

def saveData(excel_filename, foot_data):
          
    # print out foot dictionary    
    good_keys = []
    good_vals = []
    
    all_feet = getAllFeet()
    
    for foot in all_feet:
        k1 = foot + '_down'
        k2 = foot + '_up'
        if k1 in foot_data.keys():
            
            print('\n')
            v1 = ' '.join([str(x) for x in foot_data[k1]])
            v2 = ' '.join([str(x) for x in foot_data[k2]])
                              
            print(foot + ' down: ' + v1)
            print(foot + ' up:   ' + v2)
            good_keys.extend([k1,k2])
            good_vals.extend([v1,v2])
             
    # save foot dictionary to excel
    d = {'leg_state':good_keys,'times':good_vals}
    df = pd.DataFrame(d)
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        df.to_excel(writer, index=False, sheet_name='steptracking')

    return

def createMovDataFile(movieFolder, videoFile, first_frame, last_frame):
    # look for a 'mov_data.txt' file in movieFolder
    # if none there, create one

    out_file = os.path.join(movieFolder, 'mov_data.txt')
    movieData = glob.glob(out_file)

    if len(movieData) == 0:
        print('No mov_data.txt file, making one ...')
        vid = cv2.VideoCapture(os.path.join(movieFolder, videoFile))
        vidlength = gaitFunctions.getVideoStats(vid, printout=True)[0]

        print('\n Writing to ' + out_file + ' .... ')
        with open(out_file, 'w') as o:
            o.write('MovieName: ' + videoFile + '\n')
            o.write('Length: ' + str(vidlength) + '\n')
            o.write('Analyzed Frames: ' + str(first_frame/1000) + '-' + str(last_frame/1000) + '\n')
            o.write('Speed Frames: ' + str(first_frame/1000) + '-' + str(last_frame/1000) + '\n')

    return out_file

def getFeetDone(foot_data):
    foot_keys = foot_data.keys()
    foot = [x.split('_')[0] for x in foot_keys]
    feet_done = list(set(foot))
    return feet_done

def getFeetToDo(feet_done):
    allFeet = getAllFeet()
    still_need = list(set(allFeet) - set(feet_done))
    # make sure they are in the right order
    feet_to_do = [x for x in allFeet if x in still_need ]
    return feet_to_do

def getAllFeet():
    feet = ['L1', 'R1', 'L2', 'R2', 'L3', 'R3', 'L4', 'R4']
    return feet

def selectFeet():
    selection = input('Enter feet to analyze (separated by spaces) or select (a)ll: ')
    if selection in ['a','all','A']:
        feet_to_do = getAllFeet()
    else:
        feet_to_do = selection.split(' ')
    return feet_to_do

def showFootDownUp(footname, footdata):
    thing = '\n'
    thing += 'Data for ' + footname + '\n'
    thing += 'Foot Down: ' + ' '.join([str(x) for x in footdata[0]]) + '\n'
    thing += 'Foot Up: ' + ' '.join([str(x) for x in footdata[1]]) + '\n'
    return thing

def filenameToTime(filename):
    t = filename.split('.')[0].split('_')[-1].lstrip('0')
    if len(t) > 0:
        return t
    else:
        return 0

def stepThroughFrames(folder_name, footname, resize=100):

    # search in this folder for .png files
    search_term = os.path.join(folder_name, '*.png')
    frames = sorted(glob.glob(search_term))

    # open up the frames in order
    numFrames = len(frames)

    i = 0

    footDown = []
    footUp = []

    current_state = ''

    print('Looking at frames')
    while True:

        if i >= numFrames:
            i = 0
            print('All done tracking ' + footname + '!')
            print('... pausing a bit before going back to beginning - (q)uit this leg? ')
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            sleep(2)
            
            # this is an opportunity to do some quality control on the downs and ups
            # quality control code . . .
            problem = gaitFunctions.qcDownsUps(footDown,footUp)
            if len(problem) > 0:
                print(problem)
                break

        frame_name = footname + ' (' + current_state + '): frame ' + str(i + 1) + ' of ' + str(numFrames) + ' ...(esc or q) when finished'
        #print('looking at ' + frames[i])

        im = cv2.imread(frames[i])
        t = float(filenameToTime(frames[i])) / 1000 # convert to seconds

        # resize if image too big for screen?
        if resize != 100:

            scale_percent = resize  # percent of original size
            width = int(im.shape[1] * scale_percent / 100)
            height = int(im.shape[0] * scale_percent / 100)
            dim = (width, height)
            im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

        cv2.namedWindow(frame_name)
        cv2.moveWindow(frame_name, 10, 10)  # Move it to top left
        cv2.imshow(frame_name, im)

        key = cv2.waitKey(0)

        if key == ord('n'):  # next image
            i += 1
            cv2.destroyAllWindows()
        elif key == ord('p'):  # previous image
            i -= 1
            cv2.destroyAllWindows()

        elif key == ord('b'):  # go to beginning
            i = 0
            cv2.destroyAllWindows()
            print('Going to beginning!')
            
        elif key == ord('e'):  # go to end
            i = numFrames-1
            cv2.destroyAllWindows()
            print('Going to end!')

        ## focus on one leg of interest and get timing of foot down and foot up
        elif key == ord('d'):  # foot down!
            # t = filenameToTime(frames[i])
            print('you pressed d = foot down!')

            if current_state == 'down':
                print('Current leg state is down ... skipping this time (' + str(t) + ')')
            else:
                current_state = 'down'
                # get this time and add it to the list for this leg
                footDown.append(t)
                # print current list of times for foot down
                print(footname + ' down: ' + ' '.join([str(x) for x in footDown]))

        elif key == ord('u'):  # foot up!
            print('you pressed u = foot up!')
            # t = filenameToTime(frames[i])
            if current_state == 'up':
                print('Current leg state is up ... skipping this time (' + str(t) + ')')
            else:
                current_state = 'up'
                # get this time and add it to the list for this leg
                footUp.append(t)
                # print current list of times for foot down
                print(footname + ' up: ' + ' '.join([str(x) for x in footUp]))
                
        elif key == ord('x'): # made a mistake and want to clear your latest entry
            if current_state == 'up':
                print('Erasing the last "up" value and reverting state to "down"')
                footUp.pop()
                current_state = 'down'
            elif current_state == 'down':
                print('Erasing the last "down" value and reverting state to "up"')
                footDown.pop()
                current_state = 'up'
            else:
                print('Ignoring your "x" - no current leg state!')
                

        elif key == 27 or key == ord('q'):  # escape or quit

            # close image window
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            ## return individual leg data
            # data = [sorted(x) for x in [R1,L1,R2,L2,R3,L3,R4,L4]]
            
            ## return foot down and foot up data for this leg
            data = [sorted(x) for x in [footDown, footUp]]
            
            # this is an opportunity to do some quality control on this data
            problem = gaitFunctions.qcDownsUps(footDown,footUp)
            if len(problem) > 0:
                print(problem)

            return data

def saveFrames(frame_folder, movie_file):
    
    # check to see if frames folder exists; if not, make a folder
    flist = glob.glob(frame_folder)

    if len(flist) == 1:
        print(' ... frames already saved for ' + movie_file + '\n')
        return frame_folder

    print('Saving frames for ' + movie_file + ' . . . . ')
    print('.... creating a directory =  ' + str(frame_folder))
    os.mkdir(frame_folder)

    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

    vid = cv2.VideoCapture(movie_file)
    fps = vid.get(5)

    print('.... saving frames!')

    base_name = movie_file.split('.')[0]
    
    frame_number = 0
    while (vid.isOpened()):
        
        ret, frame = vid.read()
        if ret: # found a frame
        
            frame_number += 1
            
            # Get frame time and save it in a variable
            # frameTime = int(vid.get(cv2.CAP_PROP_POS_MSEC))
            frameTime = round(float(frame_number)/fps,4)

            # put the time variable on the video frame
            frame = cv2.putText(frame, str(frameTime),
                                (100, 100),
                                font, 1,
                                (55, 55, 55),
                                4, cv2.LINE_8)

            # save frame to file, with frameTime
            if frameTime > 0: # cv2 sometimes(?) assigns the last frame of the movie to time 0            
                file_name = base_name + '_' + str(int(frameTime*1000)).zfill(5) + '.png'
                cv2.imwrite(os.path.join(frame_folder, file_name), frame)
            
        else: # no frame here
            break
    vid.release()
    return frame_folder

def displayFrame(frame):
    # show the frame
    cv2.imshow('press any key to exit', frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        return image

if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
        try:
            resize = int(sys.argv[2])
        except:
            resize = 100
    else:
        movie_file = gaitFunctions.select_movie_file()
        resize = 100

    print('Resizing to ' + str(resize) + '%')
    main(movie_file, resize)
