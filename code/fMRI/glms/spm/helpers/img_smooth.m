function img_smooth(img_path, kernel)

    % Smooth an image using SPM
    %
    % Arguments
    % ---------
    % img_path : str
    % kernel : numeric
    %   fwhm of gaussian smooth kernel
    %

    addpath /hpc/packages/minerva-centos7/spm/spm12 % for minerva
    
    f = filesep;
    [output_dir, img_fname, ~] = fileparts(img_path(1,:));
    edit_fname   = [output_dir f 's_' img_fname '.nii'];
    output_fname = [output_dir f img_fname '_smoothed' num2str(kernel) '.nii'];
        
    if ~isfile(output_fname)

        spm('defaults', 'FMRI');
        disp('Smoothing image')
        batch{1}.spm.spatial.smooth.data = cellstr(img_path);
        batch{1}.spm.spatial.smooth.fwhm = [kernel kernel kernel];
        batch{1}.spm.spatial.smooth.dtype = 0;
        batch{1}.spm.spatial.smooth.im = 0;
        batch{1}.spm.spatial.smooth.prefix = 's_';
        spm_jobman('run', batch)

        movefile(edit_fname, output_fname) % change name
        
    else
        disp('Already smoothed')
    end
end
