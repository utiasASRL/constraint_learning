stereo:
	papermill -p lifter_type "stereo" -p d 1 -p level no -p save 1 ConstraintsLearning.ipynb Stereo1D.ipynb
	papermill -p lifter_type "stereo" -p d 2 -p level no -p save 1 ConstraintsLearning.ipynb Stereo2D_no.ipynb
	papermill -p lifter_type "stereo" -p d 2 -p level urT -p save 1 ConstraintsLearning.ipynb Stereo2D_urT.ipynb
	papermill -p lifter_type "stereo" -p d 3 -p level no -p save 1 ConstraintsLearning.ipynb Stereo3D_no.ipynb
	papermill -p lifter_type "stereo" -p d 3 -p level urT -p save 1 ConstraintsLearning.ipynb Stereo3D_urT.ipynb
	jupyter nbconvert --to pdf Stereo*.ipynb
	rm Stereo*.ipynb
