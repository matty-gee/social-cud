function img_resample(source_path, ref_path)

    % Resample an image to a template using SPM
    %
    % Arguments
    % ---------
    % source_path : str
    % ref_path : str
    
    addpath /hpc/packages/minerva-centos7/spm/spm12 % for minerva

    % coregister & reslice with SPM
    spm('defaults', 'FMRI');
    batch{1}.spm.spatial.coreg.estwrite.ref = cellstr(ref_path);
    batch{1}.spm.spatial.coreg.estwrite.source = cellstr(source_path);
    batch{1}.spm.spatial.coreg.estwrite.other = {''};
    batch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
    batch{1}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
    batch{1}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    batch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
    batch{1}.spm.spatial.coreg.estwrite.roptions.interp = 4;
    batch{1}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
    batch{1}.spm.spatial.coreg.estwrite.roptions.mask = 0;
    batch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';
    spm_jobman('run', batch)
    
    % rename & save
    [output_dir, img_fname, ~] = fileparts(source_path);
    edit_fname   = [output_dir '/r' img_fname '.nii'];
    output_fname = [output_dir '/' img_fname '_resampled.nii'];
    movefile(edit_fname, output_fname)