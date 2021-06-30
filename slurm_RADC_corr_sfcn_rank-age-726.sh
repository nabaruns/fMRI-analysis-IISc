#!/bin/sh
#SBATCH --job-name=radc_corr4    # Job name
#SBATCH --array=1-100%3
#SBATCH --ntasks=1         # Run on a single CPU
#SBATCH --time=04:00:00  # Time limit hrs:min:sec
#SBATCH -o log/%x-%A-%a.out
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_all_4G
##SBATCH --partition=q2h_12h-32C
pwd; hostname; date |tee result

SCRATCH_DIR="/home/scratch/nabaruns"
BASE_DIR="/media/nabarun/TATA_MRI_Data_RAW"
BIDS_DIR="$SCRATCH_DIR/incorr"
OUT_DIR="$SCRATCH_DIR/outcorr"

cmd="python3 $HOME/slurm_RADC_corr_sfcn_rank-age-726.py $SLURM_ARRAY_TASK_ID"
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

# rm -rf $BIDS_DIR/$subject/

pwd; hostname; date |tee result
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode