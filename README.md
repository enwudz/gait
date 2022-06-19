# gait
 tardigrade gait analysis

Make a folder for each video

Watch video decide on clips to extract (duration of 10 sec or greater)
Make a folder for each clip, and save clips to folders

Run frame_stepper.py

    First, it asks which folder to find the movie in
        in this folder, it creates a mov_info.txt file if there is not one already
        in mov_info.txt, it puts some info, e.g. movie name, movie length

    Then, it will ask you which LEG you want to focus on / track

        you can step through the video:
            (n)ext frame or (p)revious frame
        for the focus leg, type the appropriate letter at the appropriate frame:
            leg (d)own or leg (u)p

        Keep track of mistakes, if any

        hit Escape when done
            if mistakes, edit the mov_info.txt file

2. Run the script save_step_data.py (be sure to run in python 3)

3. See jupyter notebooks for additional analysis (in order)
	average_step_plots.ipynb
	leg_swing_groups.ipynb

4. To compare different experiments, see
	compare_step_parameters.ipynb