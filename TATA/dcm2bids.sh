#!/bin/sh
subj=$1
sub_dir=$( echo $subj | sed "s/[ _]//g" )
BASE_DIR="/media/nabarun/TATA_MRI_Data_RAW"
DATA_DIR="$BASE_DIR/TATA_MRI_Baseline_18_Feb_2020/$subj/DICOM"
BIDS_DIR="$HOME/TATA/BIDS/$sub_dir"
FMRIPREP_DIR="$HOME/TATA/fmriprep/$sub_dir"
mkdir -p $BIDS_DIR
mkdir -p $FMRIPREP_DIR
dcm2bids_scaffold -o $BIDS_DIR
echo "study imaging data for $subj"
cmd="dcm2bids -d $DATA_DIR -p $sub_dir -c $HOME/TATA/config_rest.json -o $BIDS_DIR --forceDcm2niix"
echo Commandline: $cmd
# eval $cmd
# exitcode=$?
# if [ "$exitcode" -ne "0" ]
# then
# 	echo "dcm2bids error"
# fi
echo Finished dcm2bids with exit code $exitcode
# cmd2="singularity run --bind $HOME/TATA:/data --cleanenv /media/nabarun/TATA_MRI_Data_RAW/my_images/fmriprep-1.4.1.simg \
#     $BIDS_DIR $FMRIPREP_DIR participant \
#     --fs-no-reconall --output-spaces MNI152NLin6Asym:res-2 \
#     --participant-label $sub_dir --fs-license-file /data/freesurfer.txt \
#     --omp-nthreads 8 --nthreads 12 --mem_mb 30000"
cmd2="sudo fmriprep-docker $BIDS_DIR $FMRIPREP_DIR --participant_label $sub_dir --fs-license-file $HOME/TATA/freesurfer.txt --fs-no-reconall --output-spaces MNI152NLin6Asym:res-2"
echo $cmd2
eval $cmd2
exitcode=$?
if [ "$exitcode" -ne "0" ]
then
	echo "fmriprep error"
fi
echo Finished fmriprep with exit code $exitcode
exit $exitcode