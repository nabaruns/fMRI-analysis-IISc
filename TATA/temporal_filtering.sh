#% fslmath -in.nii -bptf <hp_sigma> -1 -out.nii) 
#% hp_sigma = para.hp/ 2*TR;
#% tr = 3;
#% para.hp =100;
#% hp_sigma =  100/6;
#cmd = sprintf('fslmaths /media/sridharlab/Common/ADNI/fMRI/roi_extraction_code/denoised_func_data_nonaggr.nii.gz -bptf 16.67 -1 /media/sridharlab/Common/ADNI/fMRI/roi_extraction_code/filtered_data.nii.gz');
#system(cmd);
# resting-state TR is 3.2
#high pass sigma value is 100/2*3.2 = 15.625

for subj in sub-*; do
	#gunzip ~/ssd/Guru/data/$subj/*.nii.gz
	fslmaths /home/fmri/ssd/Guru/data/$subj/func/*.nii.gz -bptf 15.625 -1 final_$subj.nii.gz
	rm /home/fmri/ssd/Guru/data/$subj/func/*.nii.gz
	echo $subj is completed
done