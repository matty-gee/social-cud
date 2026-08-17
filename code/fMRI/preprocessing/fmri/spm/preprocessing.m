clear

spm_jobman('initcfg');
spm('defaults', 'FMRI');
spm_dir = '/Users/matty_gee/Documents/MATLAB/spm/';
if isempty(cellstr(spm_select('FPList', [spm_dir '/tpm'], 'TPM.nii')))
    error(['Cannot find a TPM image in ' spm_dir])
end

%% Parameters - PROJECT SPECIFIC
    % to read dicom use dicominfo('dicom.dcm')

% slice order is in ms for multiband data
slice_order = [0;685;392.5;97.5;782.5;490.00000001;195.00000001;882.50000001;587.49999999;292.5; 
               0;685;392.5;97.5;782.5;490.00000001;195.00000001;882.50000001;587.49999999;292.5;
               0;685;392.5;97.5;782.5;490.00000001;195.00000001;882.50000001;587.49999999;292.5;
               0;685;392.5;97.5;782.5;490.00000001;195.00000001;882.50000001;587.49999999;292.5;
               0;685;392.5;97.5;782.5;490.00000001;195.00000001;882.50000001;587.49999999;292.5;
               0;685;392.5;97.5;782.5;490.00000001;195.00000001;882.50000001;587.49999999;292.5;
               0;685;392.5;97.5;782.5;490.00000001;195.00000001;882.50000001;587.49999999;292.5;];
nslices    = length(slice_order);
ref_slice  = slice_order(nslices/2); % near middle of TR (in ms)
vox        = [2.1 2.1 2.1];
tr         = 1;
ta         = tr-(tr/nslices);

%% Subjects
subdir = '/Users/matty_gee/Desktop/SocialDysfunction/data/preprocessed/fmri';
subs   = cellstr(spm_select('List', subdir, 'dir'));

%% Preprocessing
for n = 2 : length(subs)
    
    % get data
    sub_path   = [subdir '/' subs{n}]; 
    func_imgs  = cellstr(spm_select('ExtFPList', [sub_path '/func/'], '^sub.*nii$'));
    struct_img = cellstr(spm_select('FPList', [sub_path '/anat/'], '^sub.*nii$')); 
    if isempty(struct_img)
        error('Cannot find a structural image in the anat folder')
    end

    % re-align & unwarp
    batch{1}.spm.spatial.realignunwarp.data.scans = func_imgs;
    batch{1}.spm.spatial.realignunwarp.data.pmscan = '';
    batch{1}.spm.spatial.realignunwarp.eoptions.quality = 0.9;
    batch{1}.spm.spatial.realignunwarp.eoptions.sep = 4;
    batch{1}.spm.spatial.realignunwarp.eoptions.fwhm = 5;
    batch{1}.spm.spatial.realignunwarp.eoptions.rtm = 0;
    batch{1}.spm.spatial.realignunwarp.eoptions.einterp = 2;
    batch{1}.spm.spatial.realignunwarp.eoptions.ewrap = [0 0 0];
    batch{1}.spm.spatial.realignunwarp.eoptions.weight = '';
    batch{1}.spm.spatial.realignunwarp.uweoptions.basfcn = [12 12];
    batch{1}.spm.spatial.realignunwarp.uweoptions.regorder = 1;
    batch{1}.spm.spatial.realignunwarp.uweoptions.lambda = 100000;
    batch{1}.spm.spatial.realignunwarp.uweoptions.jm = 0;
    batch{1}.spm.spatial.realignunwarp.uweoptions.fot = [4 5];
    batch{1}.spm.spatial.realignunwarp.uweoptions.sot = [];
    batch{1}.spm.spatial.realignunwarp.uweoptions.uwfwhm = 4;
    batch{1}.spm.spatial.realignunwarp.uweoptions.rem = 1;
    batch{1}.spm.spatial.realignunwarp.uweoptions.noi = 5;
    batch{1}.spm.spatial.realignunwarp.uweoptions.expround = 'Average';
    batch{1}.spm.spatial.realignunwarp.uwroptions.uwwhich = [2 1];
    batch{1}.spm.spatial.realignunwarp.uwroptions.rinterp = 4;
    batch{1}.spm.spatial.realignunwarp.uwroptions.wrap = [0 0 0];
    batch{1}.spm.spatial.realignunwarp.uwroptions.mask = 1;
    batch{1}.spm.spatial.realignunwarp.uwroptions.prefix = 'u';

    % slice time correction
    batch{2}.spm.temporal.st.scans{1}(1) = cfg_dep('Realign & Unwarp: Unwarped Images (Sess 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{1}, '.','uwrfiles'));
    batch{2}.spm.temporal.st.nslices = nslices;
    batch{2}.spm.temporal.st.tr = tr;
    batch{2}.spm.temporal.st.ta = ta;
    batch{2}.spm.temporal.st.so = slice_order;
    batch{2}.spm.temporal.st.refslice = ref_slice;
    batch{2}.spm.temporal.st.prefix = 'a';

    % co-registration
        %ref img: mean img from realignment & unwarping 
        %souce img: moved to match the ref - t1
        %other imgs: remain in alignment with source img (same transformation) - realigned & slice time corrected t2*
    batch{3}.spm.spatial.coreg.estimate.ref(1) = cfg_dep('Realign & Unwarp: Unwarped Mean Image', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','meanuwr'));
    batch{3}.spm.spatial.coreg.estimate.source = struct_img;
    batch{3}.spm.spatial.coreg.estimate.other(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 1)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
    batch{3}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    batch{3}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
    batch{3}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    batch{3}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];

    % segmentation: 1=grey matter, 2=white matter, 3=csf
        % prefixes:
            % c=native space 
            % m=modulated
            % w=warped
            % r=for dartel toolbox
    batch{4}.spm.spatial.preproc.channel.vols(1) = struct_img;
    batch{4}.spm.spatial.preproc.channel.biasreg = 0.001;
    batch{4}.spm.spatial.preproc.channel.biasfwhm = 60;
    batch{4}.spm.spatial.preproc.channel.write = [0 1];
    batch{4}.spm.spatial.preproc.tissue(1).tpm = {[spm_dir '/tpm/TPM.nii,1']};
    batch{4}.spm.spatial.preproc.tissue(1).ngaus = 1;
    batch{4}.spm.spatial.preproc.tissue(1).native = [1 1];
    batch{4}.spm.spatial.preproc.tissue(1).warped = [1 1];
    batch{4}.spm.spatial.preproc.tissue(2).tpm = {[spm_dir '/tpm/TPM.nii,2']};
    batch{4}.spm.spatial.preproc.tissue(2).ngaus = 1;
    batch{4}.spm.spatial.preproc.tissue(2).native = [1 1];
    batch{4}.spm.spatial.preproc.tissue(2).warped = [1 1];
    batch{4}.spm.spatial.preproc.tissue(3).tpm = {[spm_dir '/tpm/TPM.nii,3']};
    batch{4}.spm.spatial.preproc.tissue(3).ngaus = 2;
    batch{4}.spm.spatial.preproc.tissue(3).native = [1 1];
    batch{4}.spm.spatial.preproc.tissue(3).warped = [1 1];
    batch{4}.spm.spatial.preproc.tissue(4).tpm = {[spm_dir '/tpm/TPM.nii,4']};
    batch{4}.spm.spatial.preproc.tissue(4).ngaus = 3;
    batch{4}.spm.spatial.preproc.tissue(4).native = [1 1];
    batch{4}.spm.spatial.preproc.tissue(4).warped = [1 1];
    batch{4}.spm.spatial.preproc.tissue(5).tpm = {[spm_dir '/tpm/TPM.nii,5']};
    batch{4}.spm.spatial.preproc.tissue(5).ngaus = 4;
    batch{4}.spm.spatial.preproc.tissue(5).native = [1 1];
    batch{4}.spm.spatial.preproc.tissue(5).warped = [1 1];
    batch{4}.spm.spatial.preproc.tissue(6).tpm = {[spm_dir '/tpm/TPM.nii,6']};
    batch{4}.spm.spatial.preproc.tissue(6).ngaus = 2;
    batch{4}.spm.spatial.preproc.tissue(6).native = [1 1];
    batch{4}.spm.spatial.preproc.tissue(6).warped = [1 1];
    batch{4}.spm.spatial.preproc.warp.mrf = 1;
    batch{4}.spm.spatial.preproc.warp.cleanup = 1;
    batch{4}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    batch{4}.spm.spatial.preproc.warp.affreg = 'mni';
    batch{4}.spm.spatial.preproc.warp.fwhm = 0;
    batch{4}.spm.spatial.preproc.warp.samp = 3;
    batch{4}.spm.spatial.preproc.warp.write = [1 1]; 

    % normalization: write (by applying segmentation forward deformation on coregistered imgs)
    batch{5}.spm.spatial.normalise.write.subj.def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
    batch{5}.spm.spatial.normalise.write.subj.resample(1) = cfg_dep('Coregister: Estimate: Coregistered Images', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','cfiles'));
    batch{5}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
                                                              78 76 85];
    batch{5}.spm.spatial.normalise.write.woptions.vox = vox; % can just be voxel size... can't achieve higher resolution, so smaller values will just increase size of files...
    batch{5}.spm.spatial.normalise.write.woptions.interp = 4;
    batch{5}.spm.spatial.normalise.write.woptions.prefix = 'w';
    
    % % smoothing: 6 mm (pretty standard in lit)
    % batch{6}.spm.spatial.smooth.data(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{5}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
    % batch{6}.spm.spatial.smooth.fwhm = [6 6 6];
    % batch{6}.spm.spatial.smooth.dtype = 0;
    % batch{6}.spm.spatial.smooth.im = 0;
    % batch{6}.spm.spatial.smooth.prefix = 's6';
    
    % Run the job 
    spm_jobman('run', batch);
end


