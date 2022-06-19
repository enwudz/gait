#!/usr/bin/python
import cv2
import sys
import os
import glob
from gait_analysis import *

# aside: to make a movie from a bunch of frames
# brew install ffmpeg
# ffmpeg below worked 19 May 2021
# ffmpeg -f image2 -r 10 -s 1080x1920 -pattern_type glob -i '*.png' -vcodec mpeg4 movie.mp4
# -r is framerate of movie

def main(resize=100):

    # in terminal, navigate to the directory that has your movies

    # find the video to analyze
    cwd = os.getcwd()
    movie_folder = selectOneFromList(listDirectories()) # from gait_analysis
    video_file = getMovieFromFileList(movie_folder)

    print('\n ... opening ' + video_file)

    # look for a 'mov_data.txt' file in movie_folder
    # if none there, create one
    createMovDataFile(movie_folder, video_file)

    # get first frame
    # f = getFirstFrame(video_file)
    # displayFrame(f)

    # look for frame folder in movie_folder
    # if none there, create one and save frames
    frame_folder = saveFrames(movie_folder, video_file)

    # get foot of interest
    footname = assignFoot()

    # step through frames and label things
    data = stepThroughFrames(frame_folder, resize) # enter number to scale video

    # print out step timing for ALL feet
    # printFootSteps(data)

    ## print out foot down and foot up data for this foot
    foot_info = showFootDownUp(footname, data)
    print(foot_info)

    ## add new data to mov_data.txt file
    with open(cwd + '/' + movie_folder + '/mov_data.txt', 'a') as o:
        o.write(foot_info)

    return

def createMovDataFile(movieFolder, videoFile):
    # look for a 'mov_data.txt' file in movieFolder
    # if none there, create one

    out_file = os.path.join(movieFolder, 'mov_data.txt')
    movieData = glob.glob(out_file)

    if len(movieData) == 0:
        print('No mov_data.txt file, making one ...')
        vid = cv2.VideoCapture(os.path.join(movieFolder, videoFile))
        vidlength = getVideoStats(vid, printout=True)[0]

        print('\n Writing to ' + out_file + ' .... ')
        with open(out_file, 'w') as o:
            o.write('MovieName: ' + videoFile + '\n')
            o.write('Length: ' + str(vidlength) + '\n')
            o.write('Speed: none\n')

    return out_file


def assignFoot():
    footname = input('Enter a foot name: ')
    return footname


def showFootDownUp(footname, footdata):
    thing = '\n'
    thing += 'Data for ' + footname + '\n'
    thing += 'Foot Down: ' + ' '.join([str(x) for x in footdata[0]]) + '\n'
    thing += 'Foot Up: ' + ' '.join([str(x) for x in footdata[1]]) + '\n'
    return thing


def printFootSteps(footdata):
    feet = ['R1', 'L1', 'R2', 'L2', 'R3', 'L3', 'R4', 'L4']
    for i, f in enumerate(feet):
        print(f, footdata[i])

def filenameToTime(filename):
    t = filename.split('.')[0].split('_')[-1].lstrip('0')
    if len(t) > 0:
        return int(t)
    else:
        return 0

def stepThroughFrames(folder_name, resize=100):

    # search in this folder for .png files
    search_term = os.path.join(folder_name, '*.png')
    frames = sorted(glob.glob(search_term))

    # open up the frames in order
    numFrames = len(frames)

    i = 0

    R1 = []
    R2 = []
    R3 = []
    R4 = []
    L1 = []
    L2 = []
    L3 = []
    L4 = []
    footDown = []
    footUp = []

    current_state = ''

    print('Looking at frames')
    while True:

        if i >= numFrames:
            i = 0
            cv2.destroyAllWindows()
            print('Going to beginning!')

        frame_name = 'frame ' + str(i + 1) + ' of ' + str(numFrames) + ' ...(esc) to quit'
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

        # leg recording keys

        ## focus on one leg of interest and get timing of foot down and foot up
        elif key == ord('d'):  # foot down!
            t = filenameToTime(frames[i])
            print('you pressed d = foot down!')

            if current_state == 'd':
                print('Current leg state is down ... skipping this time (' + str(t) + ')')
            else:
                current_state = 'd'
                # get this time and add it to the list for this leg
                footDown.append(t)
                # print current list of times for foot down
                print(footDown)

        elif key == ord('u'):  # foot up!
            print('you pressed u = foot up!')
            t = filenameToTime(frames[i])
            if current_state == 'u':
                print('Current leg state is up ... skipping this time (' + str(t) + ')')
            else:
                current_state = 'u'
                # get this time and add it to the list for this leg
                footUp.append(t)
                # print current list of times for foot down
                print(footUp)
                
        ## get boundaries of frames where no camera motion and tardigrade walking straight
        elif key == ord('f'): # first frame
            print('you pressed f = first frame for speed')
            print('Time is ' + str(filenameToTime(frames[i])))
            # need to figure out how to save image in the right folder
            
        ## timing of each individual leg
        ## this is an older method, should use above (up and down for each leg one at a time)
        elif key == ord('1'):  # front right
            print('you pressed 1 = front right (R1)')
            # get this time and add it to the list for this leg
            R1.append(filenameToTime(frames[i]))
            # print current list of times for this leg
            print(R1)

        elif key == ord('2'):  # front left
            print('you pressed 2 = front left (L1)')
            # get this time and add it to the list for this leg
            L1.append(filenameToTime(frames[i]))
            # print current list of times for this leg
            print(L1)

        elif key == ord('q'):  # second right
            print('you pressed q = R2')
            # get this time and add it to the list for this leg
            R2.append(filenameToTime(frames[i]))
            # print current list of times for this leg
            print(R2)

        elif key == ord('w'):  # second left
            print('you pressed w = L2')
            # get this time and add it to the list for this leg
            L2.append(filenameToTime(frames[i]))
            # print current list of times for this leg
            print(L2)

        elif key == ord('a'):  # third right
            print('you pressed a = R3')
            # get this time and add it to the list for this leg
            R3.append(filenameToTime(frames[i]))
            # print current list of times for this leg
            print(R3)

        elif key == ord('s'):  # third left
            print('you pressed s = L3')
            # get this time and add it to the list for this leg
            L3.append(filenameToTime(frames[i]))
            # print current list of times for this leg
            print(L3)

        elif key == ord('z'):  # third right
            print('you pressed z = Right Rear (R4)')
            # get this time and add it to the list for this leg
            R4.append(filenameToTime(frames[i]))
            # print current list of times for this leg
            print(R4)

        elif key == ord('x'):  # third left
            print('you pressed x = Left Rear (L4)')
            # get this time and add it to the list for this leg
            L4.append(filenameToTime(frames[i]))
            # print current list of times for this leg
            print(L4)

        elif key == 27:  # escape

            ## return individual leg data
            # data = [sorted(x) for x in [R1,L1,R2,L2,R3,L3,R4,L4]]

            ## return foot down and foot up data
            data = [sorted(x) for x in [footDown, footUp]]

            return data


def saveFrames(movieFolder, videofile):
    # check to see if frames folder exists; if not, make a folder
    base_name = videofile.split('.')[0]
    dest_folder = base_name + '_frames'

    folder_name = os.path.join(base_name, dest_folder)

    flist = glob.glob(folder_name)

    if len(flist) == 1:
        print(' ... frames already saved for ' + videofile + '\n')
        return folder_name

    print('Saving frames for ' + videofile + ' . . . . ')
    print('.... creating a directory =  ' + str(folder_name))
    os.mkdir(folder_name)

    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

    vid = cv2.VideoCapture(movieFolder + '/' + videofile)

    print('.... saving frames!')
    
    while (vid.isOpened()):
        
        ret, frame = vid.read()
        if ret: # found a frame
            
            # Get frame time and save it in a variable
            frameTime = int(vid.get(cv2.CAP_PROP_POS_MSEC))

            # put the time variable over the video frame
            frame = cv2.putText(frame, str(frameTime / 1000),
                                (100, 100),
                                font, 1,
                                (55, 55, 55),
                                4, cv2.LINE_8)

            # save frame to file, with frameTime
            if frameTime > 0: # cv2 sometimes(?) assigns the last frame of the movie to time 0            
                file_name = base_name + '_' + str(frameTime).zfill(8) + '.png'
                cv2.imwrite(os.path.join(folder_name, file_name), frame)
            
        else: # no frame here
            break
    vid.release()

    return folder_name


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
