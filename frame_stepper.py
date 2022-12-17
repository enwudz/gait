#!/usr/bin/python
import cv2
import sys
import os
import glob
import gait_analysis

# aside: want to make a movie from a bunch of frames?
# brew install ffmpeg
# ffmpeg below works Oct 2022
# ffmpeg -f image2 -r 30 -pattern_type glob -i '*_frames_*.png' -pix_fmt yuv420p -crf 20 demo_movie.mp4
# -r is framerate of movie

'''
WISH LIST

1. allow to change mind . . . e.g. hit 'x' if want to clear current entry
'''

def main(resize=100):

    # in terminal, navigate to the directory that has your movies

    # find the video to analyze
    cwd = os.getcwd()
    movie_folder = gait_analysis.selectOneFromList(gait_analysis.listDirectories()) # from gait_analysis
    video_file = gait_analysis.getMovieFromFileList(movie_folder)
    print('\n ... opening ' + video_file)

    # get first frame
    # f = getFirstFrame(video_file)
    # displayFrame(f)

    # look for frame folder in movie_folder
    # if none there, create one and save frames
    frame_folder, first_frame, last_frame = saveFrames(movie_folder, video_file)

    # look for a 'mov_data.txt' file in movie_folder
    # if none there, create one
    createMovDataFile(movie_folder, video_file, first_frame, last_frame)

    # get foot of interest
    feet_to_do = getFeet()

    for footname in feet_to_do:

        print('... record data for ' + footname + '\n')

        # step through frames and label things
        # can enter a number for resize to scale video
        data = stepThroughFrames(frame_folder, footname, resize) 

        ## print out foot down and foot up data for this foot
        foot_info = showFootDownUp(footname, data)
        print(foot_info)

        ## add new data to mov_data.txt file
        with open(cwd + '/' + movie_folder + '/mov_data.txt', 'a') as o:
            o.write(foot_info)

    # if footname is R4, ask if we should run plot_steps and measurements
    # (plot steps also asks if we should remove the frame folder)
    if footname == 'R4':

        selection = input ('\nRun plot_steps.py? (y) or (n): ')
        if selection == 'y':
            import plot_steps
            plot_steps.main(movie_folder)

        selection = input ('Measure stuff? (y) or (n): ')
        if selection == 'y':
            import measure_things
            measure_things.main(movie_folder)

    return

def createMovDataFile(movieFolder, videoFile, first_frame, last_frame):
    # look for a 'mov_data.txt' file in movieFolder
    # if none there, create one

    out_file = os.path.join(movieFolder, 'mov_data.txt')
    movieData = glob.glob(out_file)

    if len(movieData) == 0:
        print('No mov_data.txt file, making one ...')
        vid = cv2.VideoCapture(os.path.join(movieFolder, videoFile))
        vidlength = gait_analysis.getVideoStats(vid, printout=True)[0]

        print('\n Writing to ' + out_file + ' .... ')
        with open(out_file, 'w') as o:
            o.write('MovieName: ' + videoFile + '\n')
            o.write('Length: ' + str(vidlength) + '\n')
            o.write('Analyzed Frames: ' + str(first_frame/1000) + '-' + str(last_frame/1000) + '\n')
            o.write('Speed Frames: ' + str(first_frame/1000) + '-' + str(last_frame/1000) + '\n')

    return out_file

def getFeet():
    feet = ['L1', 'R1', 'L2', 'R2', 'L3', 'R3', 'L4', 'R4']
    selection = input('Enter feet to analyze (separated by spaces) or select (a)ll: ')
    if selection in ['a','all','A']:
        feet_to_do = feet
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
        return int(t)
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
            print('All done with this clip - going back to beginning!')
            cv2.waitKey(50)
            cv2.destroyAllWindows()
            
            # this is an opportunity to do some quality control on the downs and ups
            # quality control code . . .
            problem = gait_analysis.qcDownsUps(footDown,footUp)
            if len(problem) > 0:
                print(problem)
                break

        frame_name = footname + ' (' + current_state + '): frame ' + str(i + 1) + ' of ' + str(numFrames) + ' ...(esc or q) when finished'
        #print('looking at ' + frames[i])

        im = cv2.imread(frames[i])
        t = filenameToTime(frames[i])

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
            t = filenameToTime(frames[i])
            print('you pressed d = foot down!')

            if current_state == 'down':
                print('Current leg state is down ... skipping this time (' + str(t) + ')')
            else:
                current_state = 'down'
                # get this time and add it to the list for this leg
                footDown.append(t)
                # print current list of times for foot down
                print('down: ' + ' '.join([str(x) for x in footDown]))

        elif key == ord('u'):  # foot up!
            print('you pressed u = foot up!')
            t = filenameToTime(frames[i])
            if current_state == 'up':
                print('Current leg state is up ... skipping this time (' + str(t) + ')')
            else:
                current_state = 'up'
                # get this time and add it to the list for this leg
                footUp.append(t)
                # print current list of times for foot down
                print('up: ' + ' '.join([str(x) for x in footUp]))
                
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
            problem = gait_analysis.qcDownsUps(footDown,footUp)
            if len(problem) > 0:
                print(problem)

            return data


def saveFrames(movieFolder, videofile):
    # check to see if frames folder exists; if not, make a folder
    base_name = videofile.split('.')[0]
    dest_folder = base_name + '_frames'

    folder_name = os.path.join(base_name, dest_folder)

    flist = glob.glob(folder_name)

    if len(flist) == 1:
        print(' ... frames already saved for ' + videofile + '\n')
        return folder_name, 0, 0 

    print('Saving frames for ' + videofile + ' . . . . ')
    print('.... creating a directory =  ' + str(folder_name))
    os.mkdir(folder_name)

    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

    vid = cv2.VideoCapture(os.path.join(movieFolder, videofile))

    print('.... saving frames!')

    frameTimes = []
    
    while (vid.isOpened()):
        
        ret, frame = vid.read()
        if ret: # found a frame
            
            # Get frame time and save it in a variable
            frameTime = int(vid.get(cv2.CAP_PROP_POS_MSEC))

            # put the time variable on the video frame
            frame = cv2.putText(frame, str(frameTime / 1000),
                                (100, 100),
                                font, 1,
                                (55, 55, 55),
                                4, cv2.LINE_8)

            # save frame to file, with frameTime
            if frameTime > 0: # cv2 sometimes(?) assigns the last frame of the movie to time 0            
                file_name = base_name + '_' + str(frameTime).zfill(8) + '.png'
                cv2.imwrite(os.path.join(folder_name, file_name), frame)
                frameTimes.append(frameTime)
            
        else: # no frame here
            break
    vid.release()
    return folder_name, frameTimes[0], frameTimes[-1]


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
        resize_factor = sys.argv[1]
        resize = int(resize_factor)
        print('resizing to ' + resize_factor + '%')
    else:
        resize = 100

    main(resize)
