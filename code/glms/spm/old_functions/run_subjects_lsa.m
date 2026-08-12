%------------------------------------------------------------------------
% Runs least squares all GLM using unsmoothed subject images
% Run prepare_data.m first
%------------------------------------------------------------------------

clear; clc;

f = filesep;
cd '/Users/matty_gee/Desktop/SocialTrajectories/code/fmri_glms'
preprc_dir = '/Users/matty_gee/Desktop/SocialTrajectories/data/preprocessed';
glm_dir    = '/Users/matty_gee/Desktop/SocialTrajectories/data/modeled/lsa_decision';
model_name = 'lsa_decision';

% timing is the same for everyone
timing = readtable('timing.xlsx'); % should be in same directory
timing = sortrows(timing, 'onset', 'ascend');

% loop over all subjects
subdirs = dir(fullfile([preprc_dir f 'fmri'],'sub*'));
for s = 1 : numel(subdirs)

    sub_id = subdirs(s).name; 
    fprintf('Now processing %d/%d: %s\n', s, numel(subdirs), sub_id);

    try
    
        %------------------- load subject data

        behavior  = readtable([preprc_dir f 'behavior' f sub_id '.xlsx']);
        behavior  = sortrows(behavior, 'decision_num', 'ascend');
        func_dir  = [preprc_dir f 'fmri' f sub_id f 'func'];
        func_imgs = cellstr(spm_select('ExtFPList', func_dir, 'func.nii'));
        rp_txt    = spm_select('FPList', func_dir, 'rp.txt');
        
        %------------------- validations

        func_imgs = cellstr(spm_select('ExtFPList', func_dir, '^func\.nii$'));
        if isempty(func_imgs)
            warning('Subject %s: no func.nii found in %s – skipping.', sub_id, func_dir);
            continue
        end
        rp_txt = spm_select('FPList', func_dir, '^rp\.txt$');
        if isempty(rp_txt)
            warning('Subject %s: no rp.txt found – skipping.', sub_id);
            continue
        end

        %------------------- define & run the glm
        
        glm = glm_make_design(model_name, timing, behavior, [glm_dir f sub_id], 0);
        glm_run(func_imgs, glm, rp_txt, [glm_dir f sub_id])

    catch ME

        warning('Subject %s failed: %s', sub_id, ME.message);
        continue

    end

end