%------------------------------------------------------------------------
% Runs parametric modulation GLM using smoothed subject images
% Run prepare_data.m first
%------------------------------------------------------------------------


clear all;
cd '/Users/matty_gee/Desktop/SocialTrajectories/code/fmri_glms'
f = filesep;

glm_name   = 'distance_delta';
preprc_dir = '/Users/matty_gee/Desktop/SocialTrajectories/data/preprocessed';
glm_dir    = ['/Users/matty_gee/Desktop/SocialTrajectories/data/modeled' f glm_name];


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
    
        behavior   = readtable([preprc_dir f 'behavior' f sub_id '.xlsx']);
        behavior   = sortrows(behavior, 'decision_num', 'ascend');
        
        % smooth data
        func_fname = 'func_smoothed6.nii';
        func_dir   = [preprc_dir f 'fmri' f sub_id f 'func'];
        if ~isfile([func_dir f func_fname]) 
            img_smooth(spm_select('ExtFPList', func_dir, 'func.nii'), 6)
        end
        func_imgs  = cellstr(spm_select('ExtFPList', func_dir, 'func_smoothed6.nii'));
        rp_txt     = spm_select('FPList', func_dir, 'rp.txt');
        
        %------------------- define & run the glm
        
        glm = glm_make_design(glm_name, timing, behavior, [glm_dir f sub_id], 0);
        glm_run(func_imgs, glm, rp_txt, [glm_dir f sub_id])

    catch ME

        warning('Subject %s failed: %s', sub_id, ME.message);
        continue

    end

end