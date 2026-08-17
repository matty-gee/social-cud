function glm_run(func_imgs, glm_cfg, nuisance_txt, glm_dir)

    f = filesep;

    % robust completion/partial run check
    spm_mat = [glm_dir f 'SPM.mat'];
    rpv_nii = [glm_dir f 'RPV.nii'];
    rpv_gz  = [glm_dir f 'RPV.nii.gz'];
    
    if isfile(spm_mat)
        if isfile(rpv_nii) || isfile(rpv_gz)
            disp('Estimation appears to have been completed already. Exiting.');
            return
        else
            disp('Unfinished old estimation. Deleting old SPM.mat');
            delete(spm_mat)
        end
    end

    % some settings are different across samples
    n_imgs = length(func_imgs);
    fprintf('Running GLM for %d images\n', n_imgs)
    if n_imgs < 1000 % initial sample
        glm_cfg.tr = 2;  % TR
        glm_cfg.mr = 36; % microtime resolution
    else % validation sample
        glm_cfg.tr = 1;
        glm_cfg.mr = 70;
    end

    
    spm('defaults', 'FMRI')

    %------------------- define design matrix    
    
    % i/o
    batch{1}.spm.stats.fmri_spec.dir              = {glm_dir}; 
    batch{1}.spm.stats.fmri_spec.sess.scans       = func_imgs; 
    
    % timing 
    batch{1}.spm.stats.fmri_spec.timing.units     = 'secs';
    batch{1}.spm.stats.fmri_spec.timing.RT        = glm_cfg.tr;
    batch{1}.spm.stats.fmri_spec.timing.fmri_t    = glm_cfg.mr;   % microtime resolution: how many time-bins to use per volume to build regressors; if stc, set to number of slices
    batch{1}.spm.stats.fmri_spec.timing.fmri_t0   = round(glm_cfg.mr/2); % microtime onset: if stc, match ref slice (probably middle slice)
    
    % condtions [optional parametric modulation]
    if isfield(glm_cfg, 'cond')
        for c = 1 : length(glm_cfg.cond)
            batch{1}.spm.stats.fmri_spec.sess.cond(c) = glm_cfg.cond(c);
        end
    else
        warning('No conditions provided in glm_cfg.cond; model may be empty.');
    end
    
    % nuisance regressors
    batch{1}.spm.stats.fmri_spec.sess.multi_reg   = {nuisance_txt};
    
    % filtering, masking etc
    batch{1}.spm.stats.fmri_spec.sess.hpf         = 128;    % high-pass filter in seconds; default = 128s (1/128 Hz); rule of thumb: threshold at 2-3x average (or max) intervals (s) between predictor onsets
    batch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];  % hemodynamic response function
    batch{1}.spm.stats.fmri_spec.mthresh          = glm_cfg.mthresh;    % -Inf: all voxels
    batch{1}.spm.stats.fmri_spec.mask             = glm_cfg.mask;   % explicit mask
    batch{1}.spm.stats.fmri_spec.cvi              = 'FAST'; % pre-whitening of serial correlations: default='AR(1)' or 'FAST'; FAST might be better at removing temporally autocorrelated BOLD signal (Olszowy et al 2018)
    
    
    %------------------- estimate glm
    
    batch{2}.spm.stats.fmri_est.spmmat(1)         = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    batch{2}.spm.stats.fmri_est.write_residuals   = glm_cfg.write_residuals;
    batch{2}.spm.stats.fmri_est.method.Classical  = 1;
    
    
    %------------------- weight contrasts
    
    
    batch{3}.spm.stats.con.spmmat                 = {[glm_dir f 'SPM.mat']};
    batch{3}.spm.stats.con.consess                = glm_cfg.consess;
    
    try
        names = cellfun(@(c) c.tcon.name, glm_cfg.consess, 'UniformOutput', false);
        fprintf('Contrasts (%d): %s\n', numel(names), strjoin(names, '; '));
    catch
    end
    
    %------------------- compute everything    
    
    try
        spm_jobman('run', batch);
    catch ME
        disp('Problem running spm_jobman. Running in interactive mode.')
        fprintf('  Identifier: %s\n', ME.identifier);
        fprintf('  Message   : %s\n', ME.message);
        spm_jobman('interactive', batch);
    end

    % Ensure estimation produced betas 
    beta_3d = spm_select('FPList', glm_dir, '^beta.*nii$');
    if isempty(beta_3d)
        warning('No beta images found after estimation in %s.', glm_dir);
    end

