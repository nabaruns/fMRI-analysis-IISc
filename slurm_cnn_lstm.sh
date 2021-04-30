#!/bin/sh
#SBATCH --job-name=crnn    # Job name
##SBATCH --array=1-608%10
#SBATCH --ntasks=1         # Run on a single CPU
#SBATCH --time=02:00:00  # Time limit hrs:min:sec
#SBATCH -o log/%x-%A-%a.out
#SBATCH --gres=gpu:1
##SBATCH --partition=q_1day-4G
##SBATCH --partition=q2h_12h-32C
#SBATCH --partition=q1m_2h-1G
pwd; hostname; date |tee result

SCRATCH_DIR="/home/scratch/nabaruns"
BASE_DIR="/media/nabarun/TATA_MRI_Data_RAW"
BIDS_DIR="$SCRATCH_DIR/incorr"
OUT_DIR="$SCRATCH_DIR/fmriprep"

cd $SCRATCH_DIR

# mkdir -p $BIDS_DIR && mkdir -p $OUT_DIR

# # subject=$( ls /scratch/nabaruns/BIDS/ | sed "${SLURM_ARRAY_TASK_ID}q;d" )
# subject=$( cat $HOME/corr_done.txt | sed "${SLURM_ARRAY_TASK_ID}q;d" )

# mkdir -p "$BIDS_DIR/$subject"

# echo "Copying $subject fmriprep"
# copy_cmd="scp -q -r nabarun@10.36.17.186:~/RADC/sym_radc/fmriprep/$subject/fmriprep/sub-*/*/*{bold.nii,timeseries,regressors}*.{tsv,gz} $BIDS_DIR/$subject"
# echo Commandline: $copy_cmd
# # scp -r nabarun@10.36.17.186:'/media/varsha/Seagate\ Backup\ Plus\ Drive/MRI_DEVARAJAR/RADC_Dicom/BIDS/$subject/*' $BIDS_DIR/$subject/
# eval $copy_cmd
# exitcode=$?
# if [ "$exitcode" -ne "0" ]
# then
# 	echo "$subject" >> $HOME/log/failed_SCP_corr.${SLURM_ARRAY_JOB_ID}
#     pwd; hostname; date |tee result
# 	echo Failed SCP tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
# 	exit $exitcode
# fi
cmd="python3 $HOME/slurm_cnn_lstm.py $OUT_DIR"
# cmd="fmriprep-docker $BIDS_DIR/$subject/ $OUT_DIR/$subject/ --participant_label $sids --fs-license-file $HOME/freesurfer.txt --fs-no-reconall --output-spaces MNI152NLin6Asym:res-2"
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

if [ "$exitcode" -ne "0" ]
then
    echo "$subject" >> $HOME/log/failed_subjects_corr.${SLURM_ARRAY_JOB_ID}
    echo "${SLURM_ARRAY_TASK_ID}" >> $HOME/log/failed_taskids_corr.${SLURM_ARRAY_JOB_ID}
# else
	# scp -r $OUT_DIR/$subject/ nabarun@10.36.17.186:"$BASE_DIR/fmriprep$TYPE_SUB/"
fi

rm -rf $BIDS_DIR/$subject/

pwd; hostname; date |tee result
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode