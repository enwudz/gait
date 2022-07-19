# gait
 tardigrade gait analysis

Record a video of walking tardigrade. We usually do videos of about 2 mins.

From this video, we will extract clips of tardigrades walking in a straight line.
Ideally, the stage would not move as the tardigrade walks a straight line, so that we can calculate speed.

Use QuickTime or another video editing app to save clips from your video.
We want clips where the tardigrade is walking in a straight line, ideally with no stage motion.
When you are finished saving clips, run 'files_to_folders.py'.
This script will put each video into its own folder.

1. Run frame_stepper.py

    First, it asks which clip you would like to analyze
        in the folder containing this clip, it creates a mov_info.txt file if there is not one already
        in mov_info.txt, it puts some info, e.g. movie name, movie length

    Then, it will ask you which LEG or LEGs you want to focus on / track

        you can enter one leg (e.g. 'L1')
        or you can enter multiple legs, separated by spaces ('L1 R1 L2 R2')
        or you can enter (a) to select all legs (it will run in order: L1 R1 L2 R2 L3 R3 L4 R4)

        for each leg, you can step through the video frame by frame:
            (n)ext frame or (p)revious frame
        for the focus leg, type the appropriate letter at the appropriate frame:
            leg (d)own or leg (u)p

        Keep track of mistakes, if any

        hit Escape when done with your leg
        
        You can edit the mov_info.txt file to correct any mistakes in timing



2. Run the script save_step_data.py (be sure to run in python 3)

3. See jupyter notebooks for additional analysis (in order)
	average_step_plots.ipynb
	leg_swing_groups.ipynb

4. To compare different experiments, see
	compare_step_parameters.ipynb