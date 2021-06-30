%%% MATLAB VERSION SHOULD BE ATLEAST 2017b or HIGHER 

addpath('/home/varsha/dMRIprep'); % Path where this script is stored

fspath = '/home/varsha/freesurfer'; %change as needed
DICOM_path = '/home/varsha/dMRIprep/data'; % Path where folders are stored with the names Subject# (eg. Subject3)

BIDS_path = '/home/varsha/dMRIprep/BIDS'; % Path to store BIDS structured folders (dcm2bids output)
dmriprep_path = '/home/varsha/dMRIprep/outputs'; %Path to fmriprep output files


%Get a list of Subjects present in the DICOM folder

Sub_list = dir(DICOM_path);
Sub_size = size(Sub_list);

% dcm2bids for all subjects

for i = 1:Sub_size(1,1) % for each subject in the DICOM folder %%CHANGE
    
    label_size = size(Sub_list(i).name);
    participant_label = Sub_list(i).name;
    
    dicom_sub_path = sprintf('%s/%s/DICOM',DICOM_path,participant_label); 
    
    % remove subject from name
    mkdir(sprintf('%s/%s',BIDS_path,participant_label)); %create subject's BIDS folder
    
    BIDS_sub_path = sprintf('%s/%s',BIDS_path,participant_label);
    mkdir(sprintf('%s/%s',dmriprep_path,participant_label)); %create subject's fmriprep output folder
    
    output_sub_path = sprintf('%s/%s',dmriprep_path,participant_label);
    
    cmd = sprintf('cd %s; dcm2bids_scaffold;',BIDS_sub_path); %CD to BIDS folder, create BIDS structure
    status = system(cmd);
    
    cmd = sprintf('cd %s;dcm2bids_helper -d %s',BIDS_sub_path,dicom_sub_path);
    status = system(cmd);
    
%     %% Find the correct rest fMRI file - largest size %%CHANGE
%     
%     nii_list = dir(sprintf('%s/tmp_dcm2bids/helper/',BIDS_sub_path));
%     
%     nii_size = size(nii_list,1);
%     dwi_files ={}; %stores info of files with the word 'RESTING' present in them
%     k=1;
%     
%     for j = 3:nii_size
%         
%         present = strfind(nii_list(j).name,'dwi');
%         if(present)
%             dwi_files{k,1} = j; %index
%             dwi_files{k,2} = nii_list(j).name; %name of file
%             dwi_files{k,3} = nii_list(j).bytes; %size of file
%             k = k+1;
%         end
%         clear present;
%     end
%     
%     [M,I] = max(cell2mat(dwi_files(:,3)));
%     config_dwi_full = char(dwi_files(I,2));
%     config_dwi = sprintf('*%s*dwi*',config_dwi_full(1,1:3));
    
    % Creating config .json file
    
    config_json_name = sprintf('%s/code/config.json',BIDS_sub_path);
    Config_sub_path = sprintf('%s/code/config.json',BIDS_sub_path);
    
    % parameters with the comment MODIFY attached are those with values only present in DICOM headers, not NIfTI.
    % Manually Modify in script as required if not using for RADC
    
    config.descriptions(1).dataType = 'dwi';
    config.descriptions(1).modalityLabel = 'dwi';
    config.descriptions(1).customLabels = 'dir-AP';
    config.descriptions(1).criteria.SidecarFilename = '007*'; %%CHANGE
    
    
    config.descriptions(2).dataType = 'anat';
    config.descriptions(2).modalityLabel = 'T1w';
    config.descriptions(2).customLabels = '';
    config.descriptions(2).criteria.SidecarFilename = '002*';%%CHANGE
    
%     config.descriptions(3).dataType = 'fmap';
%     config.descriptions(3).modalityLabel = 'epi';
%     config.descriptions(3).customLabels = 'dir-PA';
%     config.descriptions(3).criteria.SidecarFilename = '024*';%%CHANGE
    
    config_json = jsonencode(config);
    
    fid = fopen(config_json_name, 'w');
    if fid == -1, error('Cannot create JSON file'); end
    fwrite(fid, config_json, 'char');
    fclose(fid);
    
    cmd = sprintf('cd %s; dcm2bids --clobber -d %s -p %s -c %s',BIDS_sub_path,dicom_sub_path,participant_label,Config_sub_path); %final dcm2bids command
    status = system(cmd);
    
%     cd (sprintf('%s',BIDS_sub_path));
%     fid = fopen(spr, 'w');
    
    % remove tmp_dcm2bids helper folder as it is not a part of the BIDS
    % structure
    
    cmd = sprintf('rm -r %s/tmp_dcm2bids',BIDS_sub_path);
    system(cmd);
    
    % fmriprep command
%     cmd = sprintf('sudo docker run -ti --rm',...
%         '-v %s:/inputs -v %s:/outputs -v %s:/freesurfer',...
%         'nipreps/dmriprep:latest /inputs /outputs participant',...
%         '--participant-label %s --ignore fieldmaps',...
%         '--fs-license-file /freesurfer/license.txt --fs-no-reconall',...
%         BIDS_sub_path,output_sub_path,fspath,participant_label);
% %     status = system(cmd);
    
end
