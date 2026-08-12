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
    batch{1}.spm.stats.fmri_spec.mthresh          = 0.5;    % -Inf: all voxels
    batch{1}.spm.stats.fmri_spec.mask             = {''};   % explicit mask
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

    % Ensure estimation produced betas before cleanup
    beta_3d = spm_select('FPList', glm_dir, '^beta.*nii$');
    if isempty(beta_3d)
        warning('No beta images found after estimation in %s.', glm_dir);
    end

    %------------------- clean up files


    % clean up residual files
    if glm_cfg.write_residuals == 1
    
        % concatenate the residuals
        res_3d = spm_select('FPList', glm_dir, '^Res_.*nii$');
        if isempty(res_3d), error('The residual images are missing'), end
        spm_file_merge(res_3d, 'residual_tmp.nii');
    
        % write as float32
        V = spm_vol(fullfile(glm_dir, 'residual_tmp.nii'));
        Y = spm_read_vols(V);
        for i = 1:numel(V)
            V(i).fname = fullfile(glm_dir, 'residual.nii');
            V(i).dt    = [16 0];  % float32
        end
        spm_write_vol_4D(V, Y);
    
        % delete temp merged and individual residuals
        delete(fullfile(glm_dir, 'residual_tmp.nii'));
        for r = 1 : length(res_3d)
            delete(strtrim(res_3d(r, :)));
        end
    
        % delete the 3D betas
        beta_3d = spm_select('FPList', glm_dir, '^beta.*nii$');
        for b = 1:size(beta_3d,1)
            delete(strtrim(beta_3d(b, :)))
        end
    end

    % clean up the least squares all files
    % - 64 conditions: narrative (1) + 63 decision trials (2:64)
    if isfield(glm_cfg, 'cond') 
        if length(glm_cfg.cond) == 64
            
            beta_3d = spm_select('FPList', glm_dir, '^beta.*nii$');
            con_3d  = spm_select('FPList', glm_dir, '^con.*nii$');
            tval_3d = spm_select('FPList', glm_dir, '^spmT.*nii$');
    
            % rename the narrative image
            movefile(strtrim(beta_3d(1, :)), fullfile(glm_dir, 'beta_narrative.nii'))
            movefile(strtrim(con_3d(1, :)), fullfile(glm_dir, 'con_narrative.nii'))
            movefile(strtrim(tval_3d(1, :)), fullfile(glm_dir, 'spmT_narrative.nii'))
    
            % combine the 3D decision images into 4D
            spm_file_merge(beta_3d(2:64, :), 'beta_decisions.nii'); 
            spm_file_merge(con_3d(2:64, :), 'con_decisions.nii'); 
            spm_file_merge(tval_3d(2:64, :), 'spmT_decisions.nii'); 
    
            % delete the 3D images
            for b = 2:size(beta_3d,1) 
                delete(strtrim(beta_3d(b, :)))
            end
            for t = 2:size(tval_3d,1)
                delete(strtrim(tval_3d(t, :)))
                delete(strtrim(con_3d(t, :)))
            end
        end
    end
    
    % compress with gzip
    nii_files = spm_select('FPList', glm_dir, '.*\.nii$');
    for i = 1 : size(nii_files, 1)
        nii_path = strtrim(nii_files(i, :));
        [status, ~] = system(['gzip -f "', nii_path, '"']);    
        if status == 1
            warning('Compression failed for: %s. Original not deleted.', nii_path);
        end
    end
end