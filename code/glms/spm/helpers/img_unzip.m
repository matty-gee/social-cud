function img_unzip(img_fpath)

    % Unzip an image, if needed
    
    addpath /hpc/packages/minerva-centos7/spm/spm12
    
    [img_dir, img_name, ~] = fileparts(img_fpath);
    
    if isempty(spm_select('FPList', img_dir, [img_name '.nii$']))
        disp('Unzipping image')
        gunzip(spm_select('FPList', img_dir, [img_name '.nii.gz$'])); 
    else
        disp('Already unzipped')
    end    
end