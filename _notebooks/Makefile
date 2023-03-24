
all:
	papermill -p lifter_type "stereo" -p d 2 -p level 0 ConstraintsLearning.ipynb Stereo1D.ipynb
	papermill -p lifter_type "stereo" -p d 2 -p level 0 ConstraintsLearning.ipynb Stereo2D_level0.ipynb
	papermill -p lifter_type "stereo" -p d 2 -p level 3 ConstraintsLearning.ipynb Stereo2D_level3.ipynb
	papermill -p lifter_type "stereo" -p d 3 -p level 0 ConstraintsLearning.ipynb Stereo3D_level0.ipynb
	papermill -p lifter_type "stereo" -p d 3 -p level 3 ConstraintsLearning.ipynb Stereo3D_level3.ipynb
	jupyter nbconvert --to pdf Stereo*.ipynb
	rm Stereo*.ipynb
