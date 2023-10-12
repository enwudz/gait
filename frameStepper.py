#!/usr/bin/python
import cv2
import sys
import os
import glob
import gaitFunctions
import pandas as pd
# from time import sleep

# aside: want to make a movie from a bunch of frames?
# brew install ffmpeg
# ffmpeg below works Oct 2022
# ffmpeg -f image2 -r 30 -pattern_type glob -i '*_frames_*.png' -pix_fmt yuv420p -crf 20 demo_movie.mp4
# -r is framerate of movie

'''
WISH LIST

'''

def main(movie_file, resize=100):
    
    ''' ******************* 
    first, need to find the frames folder(s) for this movie
      or make these folders if they do not exist yet
    ******************* '''

    have_frame_folders = False

    # look for rotated frames folder(s) for this movie
    base_name = movie_file.split('.')[0]
    looking = '\n... Looking for rotated frames for ' + movie_file
    frame_folder_names = base_name + '*_rotacrop'
    frame_folder_list = glob.glob(frame_folder_names)
    if len(frame_folder_list) > 0:
        print(looking + ' ... found!')
        have_frame_folders = True
    else:
        print(looking + ' ... none found!')
    
    # if no rotated frames folders, look for regular frames folder(s)
    if have_frame_folders == False:
        looking = '... Looking for unprocessed frames for ' + movie_file
        frame_folder_names = base_name + '*_frames'
        frame_folder_list = glob.glob(frame_folder_names)
        if len(frame_folder_list) > 0:
            print(looking + ' ... found!')
            have_frame_folders = True
        else:
            print(looking + ' ... none found!')
    
    # if no regular frames folder(s) ... need to save some frames
    if have_frame_folders == False:
        
        print('\nWe need to save some frames to track from ' + movie_file)
        
        # find the excel file, and load the path_stats page
        path_stats = gaitFunctions.loadPathStats(movie_file)
        
        # if no path_stats, prompt to run analyzeTrack, and exit
        if len(path_stats) == 0:
            exit('No path found for this movie, run autoTracker.py and analyzeTrack.py')
            
        # print informmation about cruising bouts for this movie
        print('...this clip has ' + str(path_stats['# cruise bouts']) + ' bouts of cruising:')

        cruise_bouts = path_stats['cruise bout timing'].split('; ')
        for bout in cruise_bouts:
            print('   ' + bout)
            
        # ask if want to save frames for whole movie, or for the individual bouts
        bout_decision = input('\nMake frames from (w)hole movie, or from the (c)ruising bouts only? ').rstrip().lower()
        frames_decision = input('\nSave (r)otated and cropped frames, or (u)nprocssed frames? ').rstrip().lower()
        
        if bout_decision == 'c':
            bout_text = 'each bout'
            save_bouts = True
        else:
            bout_text = 'the whole movie'
            save_bouts = False
            
        if frames_decision == 'r':
            frames_text = 'rotated and cropped'
            rotated_frames = True
            import rotaZoomer
        else:
            frames_text = ''
            rotated_frames = False
        
        print('... OK, making a folder of ' + frames_text + ' frames for ' + bout_text)
        
        ''' *******************
        NOW we are ready to save some frames!
        ******************* '''
        if save_bouts: # save multiple bouts
            frame_folder_list = []
            for bout in cruise_bouts:
                bout_timing = bout.split(':')[1]
                boutstart = float(bout_timing.split('-')[0].replace(' ',''))
                boutend = float(bout_timing.split('-')[1].replace(' ',''))
                time_string = str(int(boutstart)) + '-' + str(int(boutend))
                
                if rotated_frames:
                    movie_clip_file = base_name + '_' + time_string
                    frame_folder = movie_clip_file + '_rotacrop'
                    rotaZoomer.main(frame_folder, movie_file, resize, 'up', boutstart, boutend)
                    print('Saving rotated and cropped frames to ' + frame_folder)
                    
                else:
                    movie_clip_file = base_name + '_' + time_string   
                    frame_folder = movie_clip_file + '_frames'
                    gaitFunctions.saveFrames(frame_folder, movie_file, True, boutstart, boutend)
                    print('Saving unprocessed frames to ' + frame_folder)
                    
                frame_folder_list.append(frame_folder)
                
        else: # saving whole movie
            if rotated_frames:
                frame_folder = base_name + '_rotacrop'
                print('Saving rotated and cropped frames to ' + frame_folder)
                rotaZoomer.main(frame_folder, movie_file, resize)
            else:
                frame_folder = base_name + '_frames'
                print('Saving frames to ' + frame_folder)
                gaitFunctions.saveFrames(movie_file)
            frame_folder_list = [frame_folder]

    
    ''' *******************
    OK now we have the frames in folders
       if more than one folder of frames available, select the ONE we want to track
    ******************* '''
    
    steptracking_sheet = 'steptracking'
    if len(frame_folder_list) > 1:
        print('Select a folder of frames to track:')
        frame_folder = gaitFunctions.selectOneFromList(frame_folder_list)
        steptracking_sheet += '_' + frame_folder.split('_')[-2]
    else:
        frame_folder = frame_folder_list[0]
    print(frame_folder)
    print(steptracking_sheet)
    
    ''' *******************
    OK, now we have the single folder of frames we want to track, 
      now get the step data dictionary and dataframe from the excel file
      if more than one folder of frames, then each folder will need (or already have) 
      a steptracking sheet in the excel file: steptracking_time-range
    ******************* '''

    foot_data, foot_data_df, excel_filename = get_foot_data(movie_file, steptracking_sheet)
    
    # get number of feet
    num_feet = gaitFunctions.get_num_feet(movie_file)

    ### ready to start tracking
    tracking = True
    all_feet = getAllFeet(num_feet)
    
    while tracking:

        # select a foot to analyze
        feet_to_do = getFeetToDo(foot_data, num_feet)
        if len(feet_to_do) > 0:
            print('\nStill need to track steps for: ' + ', '.join(feet_to_do))
            foot = select_a_foot(feet_to_do)
            if foot == 'none':
                finished_message()
                break
        else:
            print(' ... Finished tracking all legs for this clip!\n')
            finished = input('All finished tracking for now? (y) or (n) : ').rstrip()
            if finished == 'y':
                finished_message()
                break
            else:
                print('\nChoose a foot to redo: \n')
                foot = select_a_foot(all_feet)
                if foot == 'none':
                    finished_message()
                    break
                else:
                    print_foot_data(foot_data, foot)
                    print('Redoing step tracking for ' + foot + '\n')
                
        # step through frames and label up and down times for this foot
        # can enter a number for resize to scale video
        print('... record data for ' + foot + '\n')
        data = stepThroughFrames(frame_folder, foot, resize) 
    
        # if no steps, data will be empty. 
        # for data, we expect a list of two lists: down_times, up_times
        if len(data[0]) == 0 and len(data[1]) == 0:
            data[0] = [0]
            lastframe = gaitFunctions.getFrameTimes(movie_file)[-1]
            data[1] = [lastframe]# length of clip
    
        # print out foot down and foot up data for this foot
        foot_step_times = showFootDownUp(foot, data)
        print(foot_step_times)
    
        # add foot data to the foot dictionary
        # data[0] is down times, data[1] is up times
        foot_data[foot+'_down'] = data[0]
        foot_data[foot+'_up'] = data[1]

        # all done with this foot, save data!
        saveData(excel_filename, steptracking_sheet, foot_data, num_feet)

    # all done tracking this foot. Check if we have data for all feet.
    feet_to_do = getFeetToDo(foot_data, num_feet)
    if len(feet_to_do) == 0:
        
        # print out data for all feet
        for foot in all_feet:
            print_foot_data(foot_data, foot)
        
        ### ask if we should run analyzeSteps
        # selection = input('\n ... all feet are tracked, (r)un analyzeSteps.py? ')
        # if selection == 'y' or selection == 'r':
        #     import analyzeSteps
        #     analyzeSteps.main(movie_file)
        # else:
        #     print('OK - done for now.')
           
    else:
        
        # print out data so far
        feet_done = getFeetDone(foot_data, num_feet)
        for foot in feet_done:
            print_foot_data(foot_data, foot)

def finished_message():
    print(' ... OK! finished for now\n')

def print_foot_data(foot_data, foot):
    down_key = foot + '_down'
    up_key = foot + '_up'
    print('Tracking data for ' + foot + ':\n')
    print('   down times: ' + ' '.join([str(x) for x in foot_data[down_key]]))
    print('   up times:   ' + ' '.join([str(x) for x in foot_data[up_key]]))
    print('\n')

def select_a_foot(foot_list):

    i = 0
    print('  ' + str(i) + ': None - finished (for now)!')
    i += 1
    for foot in foot_list:
        print('  ' + str(i) + ': ' + foot)
        i += 1
    entry = input('\nWhich foot would you like to track? ')
    
    try:
        choice = int(entry)
    except:
        choice = int(i)

    if choice > len(foot_list) or choice == 0:
        selection = 'none'
    else:
        selection = foot_list[choice-1]
    
    return selection
        
def get_foot_data(movie_file, steptracking_sheet):
    
    ## make a dictionary to keep leg-up and leg-down times for each leg
    # keys = leg_state (like 'L1_up')
    # values = list of times (in seconds)
    foot_data = {}
    
    # load excel file for this clip, and get the step data if already present
    excel_file_exists, excel_filename = gaitFunctions.check_for_excel(movie_file)
    
    if excel_file_exists:
        
        # check if there is any step data already; load if so
        try:
            foot_data_df = pd.read_excel(excel_filename, sheet_name=steptracking_sheet, index_col=None)
        
        # if no step data, make an empty data frame to store the step data
        except:
            foot_data_df = pd.DataFrame(foot_data)        
    
    # if no excel file yet, make the excel file, and
    # make an empty data frame to store the step data
    else:
        import initializeClip
        initializeClip.main(movie_file)
        foot_data_df = pd.DataFrame(foot_data)  
    
    # if there is already step data, convert from dataframe to the foot_data dictionary
    if 'times' in foot_data_df.columns:
        
        # load foot_data dictionary
        foot_data = dict(zip(foot_data_df['leg_state'].values,foot_data_df['times'].values))
        
        # convert foot_data from string to list, to match data collection below
        for leg_state in foot_data.keys():
            if type(foot_data[leg_state]) is list:
                foot_data[leg_state] = foot_data[leg_state].split(' ')
            else:
                foot_data[leg_state] = [ foot_data[leg_state] ]
    
    return foot_data, foot_data_df, excel_filename

def saveData(excel_filename, steptracking_sheet, foot_data, num_feet, printme = False):
          
    # print out foot dictionary    
    good_keys = []
    good_vals = []
    
    all_feet = getAllFeet(num_feet)
    print('Saving step data in the steptracking tab of ' + excel_filename)
    
    for foot in all_feet:
        k1 = foot + '_down'
        k2 = foot + '_up'
        if k1 in foot_data.keys():
            
            v1 = ' '.join([str(x) for x in foot_data[k1]])
            v2 = ' '.join([str(x) for x in foot_data[k2]])
            
            if printme: 
                print('\n')                  
                print(foot + ' down: ' + v1)
                print(foot + ' up:   ' + v2)
            good_keys.extend([k1,k2])
            good_vals.extend([v1,v2])
             
    # save foot dictionary to excel
    d = {'leg_state':good_keys,'times':good_vals}
    df = pd.DataFrame(d)
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        df.to_excel(writer, index=False, sheet_name=steptracking_sheet)

    return

def getFeetDone(foot_data, num_feet):
    allFeet = getAllFeet(num_feet)
    foot_keys = foot_data.keys()
    foot = [x.split('_')[0] for x in foot_keys]
    feet_done = list(set(foot))
    feet_done = [x for x in allFeet if x in feet_done] # standardize the order
    return feet_done

def getFeetToDo(foot_data, num_feet):
    feet_done = getFeetDone(foot_data, num_feet)
    allFeet = getAllFeet(num_feet)
    still_need = list(set(allFeet) - set(feet_done))
    # make sure they are in the right order
    feet_to_do = [x for x in allFeet if x in still_need ]
    return feet_to_do

def getAllFeet(num_feet):
    feet = gaitFunctions.get_leg_list(num_feet)
    return feet

def selectFeet(num_feet):
    selection = input('Enter feet to analyze (separated by spaces) or select (a)ll: ')
    if selection in ['a','all','A']:
        feet_to_do = getAllFeet(num_feet)
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

    # initialize parameters and empty containers
    i = 0
    footDown = []
    footUp = []
    current_state = ''
    end_message = 'End of clip - press (q) or (esc) ... or go to the (b)eginning!'
    # frame_name = ''

    print('Select [e.g. click into] the image window and step through frames')
    print('... (n)ext frame, (p)revious frame, (b)eginning, (e)nd, (q)uit')
    print('... (d)own step, (u)p step, (x) = clear most recent step entry')
    while True:
  
        if i >= numFrames:
            i = numFrames-1
            print('All done tracking ' + footname + '!')
            cv2.waitKey(1)
            cv2.destroyAllWindows()       
            print(end_message)
        else:
            frame_name = footname + ' (' + current_state + '): frame ' + str(i + 1) + ' of ' + str(numFrames) + ' ...(esc or q) when finished'

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
            
            ## return foot down and foot up data for this leg
            data = [sorted(x) for x in [footDown, footUp]]
            
            # this is an opportunity to do some quality control on this data
            problem = gaitFunctions.qcDownsUps(footDown,footUp)
            if len(problem) > 0:
                print(problem)
                
            return data


if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
        try:
            resize = int(sys.argv[2])
        except:
            resize = 100
    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])
        resize = 100

    print('Resizing to ' + str(resize) + '%')
    main(movie_file, resize)
