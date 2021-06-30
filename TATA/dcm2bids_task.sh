#!/bin/sh
for d in /media/nabarun/TATA_MRI_Data_RAW/TATA_MRI_*; do
	for subj in $( ls $d ); do
		sub_dir=$( echo $subj | sed "s/[ +_]//g" )
		BASE_DIR="/media/nabarun/TATA_MRI_Data_RAW"
		DATA_DIR="$d/$subj/DICOM"
		BIDS_DIR="$HOME/TATA/BIDS_task/$sub_dir"
		FMRIPREP_DIR="$HOME/TATA/fmriprep/$sub_dir"
		mkdir -p $BIDS_DIR
		# mkdir -p $FMRIPREP_DIR
		cd $BIDS_DIR
		dcm2bids_scaffold -o $BIDS_DIR
		dcm2bids_helper -d $DATA_DIR
		funcinit=$( ls -S $BIDS_DIR/tmp_dcm2bids/helper/ | grep COGLAB | head -1 | sed "s/[_\.\-][0-9A-Za-z]*//g" )
		echo '{"descriptions":[{"dataType": "anat","modalityLabel":"T1w","criteria": {"SidecarFilename": "*TA_5.12*"}},{"dataType": "func","modalityLabel":"bold","customLabels":"task-rest","criteria": {"SidecarFilename": "'$funcinit'**COGLAB*"}}]}' > $BIDS_DIR/code/config.json
		rm -rf $BIDS_DIR/tmp_dcm2bids
		cat $BIDS_DIR/code/config.json
		echo "study imaging data for $subj"
		cmd="dcm2bids -d $DATA_DIR -p $sub_dir -c $BIDS_DIR/code/config.json -o $BIDS_DIR --forceDcm2niix"
		echo Commandline: $cmd
		eval $cmd
		exitcode=$?
		if [ "$exitcode" -ne "0" ]
		then
			echo "$subj" >> $HOME/TATA/log/failed_dcm2bids_subjects
		fi
		rm -rf $BIDS_DIR/tmp_dcm2bids
		cp -r $BIDS_DIR $BASE_DIR/BIDS_task/
		rm -rf $BIDS_DIR
		# echo Finished dcm2bids with exit code $exitcode
	done
done

# cmd2="singularity run --bind $HOME/TATA:/data --cleanenv /media/nabarun/TATA_MRI_Data_RAW/my_images/fmriprep-1.4.1.simg \
#     $BIDS_DIR $FMRIPREP_DIR participant \
#     --fs-no-reconall --output-spaces MNI152NLin6Asym:res-2 \
#     --participant-label $sub_dir --fs-license-file /data/freesurfer.txt \
#     --omp-nthreads 8 --nthreads 12 --mem_mb 30000"
# echo $cmd2
# eval $cmd2
# exitcode=$?
# if [ "$exitcode" -ne "0" ]
# then
# 	echo "fmriprep error"
# fi
# echo Finished fmriprep with exit code $exitcode
exit $exitcode