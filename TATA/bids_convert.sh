# Author: Guru
# guru-prasath@outlook.com

# convert the DICOM into BIDS format of the rest and task data
# choose the config file 'config_rest.json' or 'config_task.json' accordingly
for sub in Subject*; do
	cd $sub
	subnum="${sub#Subject}"
	dcm2bids -d /home/fmri/ssd/Guru/rest/Data/Subject$subnum/DICOM -p $subnum -o /home/fmri/ssd/Guru/rest/BIDS -c /home/fmri/ssd/Guru/rest/BIDS/code/config_rest.json
	echo $sub is completed
	cd ..
done
rm tmp_dcm2bids