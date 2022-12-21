# gait
 tardigrade gait analysis

Record a video of a moving critter. We usually do videos of about 2 mins.

From this video, we will extract clips of critters walking on a stationary background.

Use QuickTime or another video editing app to save clips from your video.

Video analysis pipeline

OK runThings.py = runs stuff below on a movie file, or on all!
	Maybe rename to pathtracker

OK initializeClip.py = makes an excel file to store all info and data from a clip
OK 6 tabs: identity, pathtracking, path_stats, steptracking, step_timing, step_stats, gait_styles

OK trackCritter.py = adds tracking data to excel file for a clip
	OK Asks which clip we want (if not provided)
	OK Checks if the excel file is there, runs initializeClip if not
	OK Checks if the tracking data is already there … asks if want to run again
	OK TRACKS, and to pathtracking tab, adds frameTime, X, Y, Area

OK analyzePath.py = analyzes the tracking data, adds stops, turns, bearings
OK Checks if the excel file is there, runs initializeClip if not, exits
OK Checks if scale available in info from identity tab
	OK If not, check for measured micrometer = '...scale.txt' file
OK If no measured micrometer, measure it
OK add scale to identity tab
	OK Reads data from pathtracking tab, analyzes to get stuff below
		OK If no data, stop and tell to run trackCritter first
OK To pathtracking tab, adds smoothedX, smoothedY, distance, cumulative distance, speed, bearing, turns(0,1), stops(0,1)
OK To path_stats tab, add summary data
	OK convert area to mm^2

OK plotPath.py = options to plot data from pathtracking tab
	OK Smoothed path (colored by time) and raw path (grey)
	OK Time vs. speed (colored by time) … with turns and stops labeled
		also include cumulative distance… on another y-axis?

OK frameStepper.py = does the feets
	OK Asks which clip we want (if not provided)
	OK Makes sure the excel file is there
	OK TRACKS, and to steptracking tab, adds the step timing
OK Quality control on up / down times
	OK Print out all step timing at end … just to have it

Need analyzeSteps.py = analyzes the step data, adds inter-step timing data
	(was save_step_data.py)
Adds this data to step_timing tab
	For each step, also get average speed over the step cycle?
Calculates summary & averages for each leg, writes this to step_stats tab

Need plotSteps.py = various options to plot steps

Need gaitStyles.py = calculate and record gait styles
Writes gait style data to gait_styles tab (or pathtracking tab?)

Need plotGaits.py = various options to plot gaits?

Need comparePaths.py = collect path data for 2(?) treatments and compares

Need compareSteps.py = collect step data for 2(?) treatments and compares

Need compareGaits.py = collect gait data for 2(?) treatments and compares