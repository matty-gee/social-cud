%------------------------------------------------------------------------
% Resting-state "Schuck-style" modeling
%
% Goal:
%   Create a resting-state series suitable for multivariate decoding / replay-style
%   analyses by applying a minimal, explicit set of transforms and using SPM's
%   serial-correlation model to obtain prewhitened residuals.
%
% Main choices:
%   1) Spatial smoothing: Gaussian smoothing at 4 mm FWHM is applied to the
%      preprocessed rest BOLD (saved as a single 4D *_s4.nii). If the smoothed 4D
%      exists and has the expected number of volumes, smoothing is skipped.
%   2) Intercept-only GLM: An SPM first-level model is fit with no task conditions
%      and no nuisance regressors (i.e., intercept only), using an explicit brain
%      mask and mthresh = -Inf (no implicit masking).
%   3) Prewhitening: Serial correlations are modeled using SPM's cvi setting
%      (FAST), which defines the prewhitening used during estimation.
%   4) High-pass filtering: SPM's HPF is effectively disabled (very large cutoff),
%      with the intent to handle trends explicitly downstream if desired.
%   5) Outputs: Only the 4D residual time series (merged from Res_#### or ResI_####)
%      and a copy of the mask are retained; all other temporary SPM outputs are
%      deleted.
%
% Assumptions:
%   - Input BOLD is already fieldmap/distortion corrected (e.g., fMRIPrep outputs).
%   - Rest TR and expected number of volumes are known (TR=1.5s, T=400).
%------------------------------------------------------------------------


clear; clc;

model_name = 'resting';

main_dir     = '/Users/matty_gee/Desktop/SocialDysfunction';
code_dir     = fullfile(main_dir, 'code', 'fmri_glms');
helpers_dir  = fullfile(code_dir, 'helpers');
preprc_dir   = fullfile(main_dir, 'data', 'preprocessed');
base_glm_dir = fullfile(main_dir, 'analyses');
glm_dir      = fullfile(base_glm_dir, model_name, 'glms');
cd(code_dir);
addpath(helpers_dir);

try
    spm('defaults','fmri'); spm_jobman('initcfg');
catch ME
    fprintf(2, 'ERROR: Failed to initialize SPM: %s\n', ME.message);
    fprintf(2, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
    return
end

expected_T  = 400;  % volumes
fwhm        = 4;    % 4 mm FWHM

% loop over all subjects
sub_dirs = dir(fullfile(preprc_dir, 'fmriprep', 'sub*'));
sub_dirs = sub_dirs(~endsWith(string({sub_dirs.name}), ".html", "IgnoreCase", true));

for s = 7 : numel(sub_dirs)

    sub_id   = sub_dirs(s).name;
    func_dir = fullfile(preprc_dir, 'fmriprep', sub_id, 'func');
    fprintf('Subject %d/%d: %s\n', s, numel(sub_dirs), sub_id);
    if ~isfolder(func_dir)
        warning('%s: func directory missing: %s', sub_id, func_dir);
        continue
    end

    % Predefine for safe cleanup in catch
    sm_path     = '';
    tmp_glm_dir = '';

    try
        sub_out_dir = fullfile(glm_dir, sub_id);
        if ~exist(sub_out_dir,'dir'), mkdir(sub_out_dir); end

        % make sure not already done
        out_res4d = fullfile(sub_out_dir, 'residuals.nii');
        out_mask  = fullfile(sub_out_dir, 'mask.nii');

        if exist(out_res4d, 'file')
            fprintf('Already done (found residuals.nii). Skipping subject: %s\n', sub_id);
            continue
        end

        % functional images
        func_base = [sub_id '_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii'];
        func_imgs = ensure_nii_and_select(func_dir, func_base);
        assert(numel(func_imgs) == expected_T, 'Expected %d volumes, found %d', expected_T, numel(func_imgs));

        % brain mask
        mask_base = [sub_id '_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii'];
        mask_img  = ensure_nii_and_select(func_dir, mask_base);

        % smooth to *_s4.nii
        sm_base = regexprep(func_base, '\.nii$', '_s4.nii');
        sm_path = fullfile(func_dir, sm_base);

        if exist(sm_path, 'file')
            try
                Vchk = spm_vol(sm_path);
                if numel(Vchk) ~= expected_T
                    warning('Smoothed file exists but has %d vols (expected %d). Rebuilding: %s', ...
                        numel(Vchk), expected_T, sm_base);
                    delete(sm_path);
                else
                    fprintf('Found smoothed 4D rest, skipping smoothing: %s\n', sm_base);
                end
            catch
                warning('Could not read existing smoothed file; rebuilding: %s', sm_base);
                try delete(sm_path); catch, end
            end
        end

        if ~exist(sm_path, 'file')
            fprintf('Smoothing rest to 4mm -> %s\n', sm_base);

            tmp_prefix = '__tmpS4_';
            tmp_base   = [tmp_prefix func_base];
            tmp_path   = fullfile(func_dir, tmp_base);
            if exist(tmp_path,'file'); try delete(tmp_path); catch, end; end

            matlabbatch = [];
            matlabbatch{1}.spm.spatial.smooth.data   = func_imgs;
            matlabbatch{1}.spm.spatial.smooth.fwhm   = [fwhm fwhm fwhm];
            matlabbatch{1}.spm.spatial.smooth.dtype  = 0;
            matlabbatch{1}.spm.spatial.smooth.im     = 0;
            matlabbatch{1}.spm.spatial.smooth.prefix = tmp_prefix;
            spm_jobman('run', matlabbatch);

            if exist(tmp_path, 'file')
                if exist(sm_path, 'file'); try delete(sm_path); catch, end; end
                [ok,msg] = movefile(tmp_path, sm_path);
                if ~ok, error('movefile failed: %s -> %s\n%s', tmp_path, sm_path, msg); end
            else
                tmp_candidates = cellstr(spm_select('FPList', func_dir, ['^' regexptranslate('escape', tmp_prefix) '.*\.nii$']));
                tmp_candidates = tmp_candidates(~cellfun(@isempty, strtrim(tmp_candidates)));
                if isempty(tmp_candidates)
                    error('Smoothing produced no tmp outputs. Expected %s or files starting with %s', tmp_base, tmp_prefix);
                end
                fprintf('Fallback: merging %d tmp files into 4D: %s\n', numel(tmp_candidates), sm_base);
                spm_file_merge(char(tmp_candidates), sm_path, 0);
                for i = 1:numel(tmp_candidates)
                    try delete(tmp_candidates{i}); catch, end
                end
            end
        end

        sm_imgs = cellstr(spm_select('ExtFPList', func_dir, ['^' regexptranslate('escape', sm_base) '$']));
        assert(numel(sm_imgs) == expected_T, 'Smoothed volume list has %d vols (expected %d).', numel(sm_imgs), expected_T);

        % intercept-only GLM (prewhiten)
        tmp_glm_dir = fullfile(sub_out_dir, '__tmp_spm_rest_prewhiten');
        if ~exist(tmp_glm_dir,'dir'), mkdir(tmp_glm_dir); end
        spm_mat = fullfile(tmp_glm_dir, 'SPM.mat');

        % ensure mask is saved
        if ~exist(out_mask,'file')
            copyfile(strip_spm_volspec(mask_img{1}), out_mask);
        end

        % compute residuals only if missing
        if ~exist(out_res4d,'file')
            fprintf('Specifying + estimating intercept-only rest GLM (tmp) -> residuals\n');

            matlabbatch = [];
            matlabbatch{1}.spm.stats.fmri_spec.dir = {tmp_glm_dir};
            matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
            matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 1.5;
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

            matlabbatch{1}.spm.stats.fmri_spec.sess.scans = sm_imgs;
            matlabbatch{1}.spm.stats.fmri_spec.sess.cond = struct([]);
            matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
            matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct([]);
            matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {''};
            matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 1e6;

            matlabbatch{1}.spm.stats.fmri_spec.mask = mask_img;
            matlabbatch{1}.spm.stats.fmri_spec.mthresh = -Inf;

            matlabbatch{1}.spm.stats.fmri_spec.cvi = 'FAST';

            matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat};
            matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
            matlabbatch{2}.spm.stats.fmri_est.write_residuals = 1;

            spm_jobman('run', matlabbatch);

            % grab residuals (SPM may write Res_####.nii or ResI_####.nii)
            res_imgs = cellstr(spm_select('FPList', tmp_glm_dir, '^Res(I)?_\d+\.nii$'));
            res_imgs = res_imgs(~cellfun(@isempty, strtrim(res_imgs)));
            res_imgs = res_imgs(cellfun(@(p) exist(p,'file')==2, res_imgs));
            if isempty(res_imgs)
                error('No residual images found in %s. Expected Res_####.nii or ResI_####.nii.', tmp_glm_dir);
            end
            assert(numel(res_imgs) == expected_T, 'Expected %d residual images, found %d', expected_T, numel(res_imgs));

            % merge to single 4D residuals.nii in sub_id folder
            fprintf('Merging %d residual volumes -> %s\n', numel(res_imgs), out_res4d);
            spm_file_merge(char(res_imgs), out_res4d, 0);
        else
            fprintf('Found residuals, skipping GLM: %s\n', out_res4d);
        end

        % cleanup: keep only residuals + mask
        try
            if exist(tmp_glm_dir,'dir')
                rmdir(tmp_glm_dir, 's');
            end
        catch ME
            warning('Could not delete temp GLM dir %s (%s).', tmp_glm_dir, ME.message);
        end

        try
            if exist(sm_path, 'file')
                delete(sm_path);
            end
        catch ME
            warning('Could not delete smoothed file %s (%s).', sm_path, ME.message);
        end

    catch ME
        % Catch *any* per-subject error (loading images, smoothing, GLM, merges, file ops, etc.)
        fprintf(2, '\nERROR processing %s: %s\n', sub_id, ME.message);
        fprintf(2, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));

        % Best-effort cleanup if temp outputs were created
        if ~isempty(tmp_glm_dir) && exist(tmp_glm_dir,'dir')
            try
                rmdir(tmp_glm_dir, 's');
            catch ME2
                warning('Could not delete temp GLM dir %s after failure (%s).', tmp_glm_dir, ME2.message);
            end
        end

        if ~isempty(sm_path) && exist(sm_path,'file')
            try
                delete(sm_path);
            catch ME2
                warning('Could not delete smoothed file %s after failure (%s).', sm_path, ME2.message);
            end
        end

        % Move on to next subject
        continue
    end

end

% ----------------------------- helper functions

function imgs = ensure_nii_and_select(func_dir, base_nii, delete_gz)
    if nargin < 3, delete_gz = true; end

    nii_path = fullfile(func_dir, base_nii);
    gz_path  = [nii_path '.gz'];

    if ~exist(nii_path, 'file')
        if ~exist(gz_path, 'file')
            error('Missing file: %s (and %s)', nii_path, gz_path);
        end

        gunzip(gz_path);

        if ~exist(nii_path, 'file')
            error('gunzip reported success but .nii not found: %s', nii_path);
        end
        d = dir(nii_path);
        if isempty(d) || d.bytes < 1024
            error('Decompressed .nii looks too small (%d bytes): %s', d.bytes, nii_path);
        end

        if delete_gz
            delete(gz_path);
        end
    end

    imgs = cellstr(spm_select('ExtFPList', func_dir, ['^' base_nii '$']));
end

function p = strip_spm_volspec(p)
% Remove trailing SPM volume specifier ",<n>" from a filename if present.
    p = char(p);
    p = regexprep(p, ',\d+$', '');
end
