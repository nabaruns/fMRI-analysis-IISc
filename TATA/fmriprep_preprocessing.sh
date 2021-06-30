# Author: Guru
# guru-prasath@outlook.com

for sub in Subject*; do
	subnum="${sub#Subject}"
	fmriprep-docker /home/fmri/ssd/Guru/rest/Subject$subnum/ /home/fmri/ssd/Guru/rest/output/ --participant_label $subnum --fs-license-file /home/fmri/ssd/Guru/fmri/freesurfer.txt --fs-no-reconall --ignore fieldmaps --output-spaces MNI152NLin6Asym:res-2
	echo $sub is completed
	cd ..
done