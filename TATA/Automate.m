addpath('/home/nabarun'); % Path where this script is stored

Freesurfer_license_path = '/home/nabarun/license.txt'; %change as needed

DICOM_path = '/home/nabarun/TATA/Data'; % Path where folders are stored with the names Subject# (eg. Subject3)

BIDS_path = '/home/nabarun/TATA/BIDS'; % Path to store BIDS structured folders (dcm2bids output)

fmriprep_path = '/home/nabarun/TATA/fmriprep'; %Path to fmriprep output files


%Get a list of Subjects present in the DICOM folder

Sub_list = dir(DICOM_path);

Sub_size = size(Sub_list);

%%dcm2bids for all subjects

for i = 3:Sub_size(1,1) % for each subject in the DICOM folder

    Label_size = size(Sub_list(i).name);
    dicom_dir = Sub_list(i).name(1:Label_size(1,2));
    Participant_label = regexprep(dicom_dir, '_', '');

    disp(Participant_label);
    
    Dicom_sub_path = sprintf('%s/%s',DICOM_path,dicom_dir); %path to the subject's DICOM folder
    
    mkdir(sprintf('%s/%s',BIDS_path,Participant_label)); %create subject's BIDS folder
    
    BIDS_sub_path = sprintf('%s/%s',BIDS_path,Participant_label);
    
    mkdir(sprintf('%s/%s',fmriprep_path,Participant_label)); %create subject's fmriprep output folder
    
    fmriprep_sub_path = sprintf('%s/%s',fmriprep_path,Participant_label);
    
    cmd = sprintf('cd %s; dcm2bids_scaffold;',BIDS_sub_path); %CD to BIDS folder, create BIDS structure
    status = system(cmd);
    
    cmd = sprintf('cd %s;dcm2bids_helper -d %s',BIDS_sub_path,Dicom_sub_path);
    status = system(cmd);
    
    %% Find the correct rest fMRI file - largest size
    
    nii_list = dir(sprintf('%s/tmp_dcm2bids/helper/',BIDS_sub_path));
    
    nii_size = size(nii_list);
    Resting_files ={}; %stores info of files with the word 'RESTING' present in them
    k=1;
    
    for j = 3:nii_size(1,1)
        
        present = strfind(nii_list(j).name,'RESTING');
        if(present)
            Resting_files{k,1} = j; %index
            Resting_files{k,2} = nii_list(j).name; %name of file
            Resting_files{k,3} = nii_list(j).bytes; %size of file
            k = k+1;
        end
        clear present;
    end
    
    [M,I] = max(cell2mat(Resting_files(:,3)));
    config_rfmri_full = char(Resting_files(I,2));
    config_rfmri =config_rfmri_full(1,1:3);
    
    %%Creating config .json file
    config_json_name = sprintf('%s/code/config.json',BIDS_sub_path);
    Config_sub_path = sprintf('%s/code/config.json',BIDS_sub_path);
    
    %parameters with the comment MODIFY attached are those with values only present in DICOM headers, not NIfTI.
    %Manually Modify in script as required if not using for RADC
    config.descriptions(1).dataType = 'func';
    config.descriptions(1).modalityLabel = 'bold';
    config.descriptions(1).customLabels = 'task-rest';
    config.descriptions(1).criteria.SidecarFilename = sprintf('%s**RESTING*',config_rfmri);
    
    
    config.descriptions(2).dataType = 'anat';
    config.descriptions(2).modalityLabel = 'T1w';
    config.descriptions(2).customLabels = 'acq-highres';
    config.descriptions(2).criteria.SidecarFilename = '*mprage**9.14*';
    
    config_json = jsonencode(config);
    
    fid = fopen(config_json_name, 'w');
    if fid == -1, error('Cannot create JSON file'); end
    fwrite(fid, config_json, 'char');
    fclose(fid);
    
    cmd = sprintf('cd %s; dcm2bids -d %s -p %s -c %s',BIDS_sub_path,Dicom_sub_path,Participant_label,Config_sub_path); %final dcm2bids command
    status = system(cmd);
    
    %remove tmp_dcm2bids helper folder as it is not a part of the BIDS structure
    
    cmd = sprintf('sudo rm -r %s/tmp_dcm2bids',BIDS_sub_path);
    status = system(cmd);
    
    %fmriprep command
    cmd = sprintf('sudo fmriprep-docker %s %s --participant_label %s --fs-license-file %s --fs-no-reconall --output-spaces MNI152NLin6Asym:res-2',BIDS_sub_path,fmriprep_sub_path,Participant_label,Freesurfer_license_path);
    status = system(cmd);
    
end
   



