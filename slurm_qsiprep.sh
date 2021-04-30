#!/bin/sh
#SBATCH --job-name=qsiprep    # Job name
#SBATCH --array=1-1%1
#SBATCH --ntasks=1         # Run on a single CPU
#SBATCH --time=02:00:00  # Time limit hrs:min:sec
#SBATCH -o log/%x-%A-%a.out
##SBATCH --gres=gpu:1
##SBATCH --partition=q_1day-4G
#SBATCH --partition=q2h_12h-32C
pwd; hostname; date |tee result

SCRATCH_DIR="/scratch/nbrn"
BIDS_DIR="$HOME/BIDSQ"
OUT_DIR="$HOME/fmriprep4Q"
WORK_DIR="$HOME/workQ"

mkdir -p $BIDS_DIR && mkdir -p $OUT_DIR && mkdir -p $WORK_DIR

# subject=$( ls /scratch/nabaruns/BIDS/ | sed "${SLURM_ARRAY_TASK_ID}q;d" )
# subject=$( cat $HOME/participants.tsv | sed "${SLURM_ARRAY_TASK_ID}q;d" )
subject="05225665_01"
sids=$( echo $subject | sed "s/_//" )
# sids=$subject

echo "Making $subject output folder"
mkdir -p $BIDS_DIR/$subject && mkdir -p $OUT_DIR/$subject && mkdir -p $WORK_DIR/$subject

echo "Copying $subject BIDS"
copy_cmd="scp -r nabarun@10.36.17.186:'/home/nabarun/RADC/BIDS/$subject' $BIDS_DIR/"
echo Commandline: $copy_cmd
# scp -r nabarun@10.36.17.186:'/media/varsha/Seagate\ Backup\ Plus\ Drive/MRI_DEVARAJAR/RADC_Dicom/BIDS/$subject/*' $BIDS_DIR/$subject/
eval $copy_cmd
exitcode=$?
if [ "$exitcode" -ne "0" ]
then
	echo "$subject" >> $HOME/log/failed_SCP_subjects.${SLURM_ARRAY_JOB_ID}
    echo "${SLURM_ARRAY_TASK_ID}" >> $HOME/log/failed_SCP_taskids.${SLURM_ARRAY_JOB_ID}
    pwd; hostname; date |tee result
	echo Failed SCP tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
	exit $exitcode
fi

cmd="docker run --init --name $SLURM_JOB_ID --user $(id -u $USER):$(id -g $USER) --rm \
	-v $BIDS_DIR/$subject/:/data \
	-v $OUT_DIR/$subject/:/out \
	-v $HOME/:/mhome/ \
	-v $WORK_DIR/$subject/:/work \
	pennbbl/qsiprep:latest \
	/data /out \
	participant \
	-w /work \
	--participant_label $sids \
	--fs-license-file /mhome/license.txt \
	--ignore fieldmaps --output-resolution 2"
	# --fs-no-reconall --low-mem --output-spaces MNI152NLin6Asym:res-2"
# cmd="fmriprep-docker $BIDS_DIR/$subject/ $OUT_DIR/$subject/ --participant_label $sids --fs-license-file $HOME/freesurfer.txt --fs-no-reconall --output-spaces MNI152NLin6Asym:res-2"
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

if [ "$exitcode" -ne "0" ]
then
    echo "$subject" >> $HOME/log/failed_subjects.${SLURM_ARRAY_JOB_ID}
    echo "${SLURM_ARRAY_TASK_ID}" >> $HOME/log/failed_taskids.${SLURM_ARRAY_JOB_ID}
else
	scp -r $OUT_DIR/$subject/ nabarun@10.36.17.186:"/home/nabarun/"
fi

rm -rf $OUT_DIR/$subject/
rm -rf $BIDS_DIR/$subject/
rm -rf $WORK_DIR/$subject/

pwd; hostname; date |tee result
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode