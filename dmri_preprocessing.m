dd = dir('*_*');
numtracks = 1000000;

for sub = 1:size(dd,1)
    
    cd (sprintf('%s',dd.name(sub)));
    
    % convert
    cmd = sprintf('mrconvert RAW/DIFFDTI45directions.nii -fslgrad RAW/*.bvec RAW/*.bval dwi.mif​');
    system(cmd);
    
    % denoise
    cmd = sprintf('dwidenoise -force -noise noise.mif dwi.mif dwi_denoised.mif​​');
    system(cmd);
    
    % Gibbs artifact removal
    cmd = sprintf('mrdegibbs -force dwi_denoised.mif dwi_degibbs.mif​​');
    system(cmd);
    
    % preprocessing
    cmd = sprintf('dwifslpreproc -quiet dwi_degibbs.mif dwi_preproc.mif -rpe_none -pe_dir ap -eddy_options "--slm=linear "​');
    system(cmd);
    
    % extract all b0 images
    cmd = sprintf('dwiextract dwi_preproc.mif -bzero b0.mif​');
    system(cmd);
    
    % average b0 images into one
    cmd = sprintf('mrmath -quiet b0.mif -axis 3 mean b0_mean.nii.gz​');
    system(cmd);
    
    % compute transform to align T1 image to DWI
    cmd = sprintf('flirt -dof 6 -cost normmi -in RAW/t1mprnssagpat2iso.nii -ref b0_mean -omat T_fsl.txt​');
    system(cmd);
    
    % brain extraction
    cmd  = sprintf('bet2 RAW/t1mprnssagpat2iso.nii T1_bet.nii.gz');
    system(cmd);
    
    % convert the transform matri thr MRtrix compatible format
    cmd = sprintf('transformconvert T_fsl.txt T1_bet.nii.gz b0_mean.nii.gz flirt_import T_T1toDWI.txt​');
    system(cmd);
    
    % apply the transform to the T1 image
    cmd = sprintf('mrtransform -force -linear T_T1toDWI.txt T1_bet.nii.gz T1_al.nii.gz');
    system(cmd);
    
    % convert the aligned T1 into .mif format. Always check visually if the
    % T1 and DWI are aligned correctly
    cmd = sprintf('mrconvert -force T1_al.nii.gz T1_al.mif​');
    system(cmd);
    
    % 5-tissue type segmentation
    cmd = sprintf('5ttgen -force -quiet fsl T1_al.mif 5TT.mif​');
    system(cmd);
    
    % extract grey-matter-white-matter interface (GMWMI)
    cmd = sprintf('5tt2gmwmi -force -quiet 5TT.mif gmwmi.mif​');
    system(cmd);
    
    % compute response function for Constrained Spherical Deconvolution
    % (CSD)
    cmd = sprintf('dwi2response tournier -force -quiet dwi_preproc.mif response.txt​');
    system(cmd);
    
    % Do the CSD and estimate Fiber Orientation Distributions (FODs)
    cmd = sprintf('dwi2fod -force -quiet csd dwi_preproc.mif response.txt WM_FODs.mif​');
    system(cmd);
    
    % whole brain tractography using GMWMI as seed
    cmd = sprintf('tckgen -force -quiet -seed_image gmwmi.mif -act 5TT.mif -backtrack -crop_at_gmwmi WM_FODs.mif whole_brain.tck -select %d', numtracks);
    system(cmd);
    
end
    