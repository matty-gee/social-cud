%------------------------------------------------------------------------
% Runs nuisance regression GLM using unsmoothed subject images
% Run prepare_data.m first
% MAYBE: also regress out time here?
%------------------------------------------------------------------------


clear all;

f = filesep;
cd '/Users/matty_gee/Desktop/SocialTrajectories/code/fmri_glms'
preprc_dir = '/Users/matty_gee/Desktop/SocialTrajectories/data/preprocessed';
glm_dir    = '/Users/matty_gee/Desktop/SocialTrajectories/data/modeled/motion-corrected';

% timing is the same for everyone
timing = readtable('timing.xlsx'); % should be in same directory
timing = sortrows(timing, 'onset', 'ascend');

% loop over all subjects
subdirs = dir(fullfile([preprc_dir f 'fmri'],'sub*'));
for s = 3:numel(subdirs)

    sub_id = subdirs(s).name; 
    fprintf('Now processing %d/%d: %s\n', s, numel(subdirs), sub_id);

    try

       %------------------- load subject data

       func_dir   = [preprc_dir f 'fmri' f sub_id f 'func'];
       func_imgs  = cellstr(spm_select('ExtFPList', func_dir, 'func.nii'));
       rp_txt     = spm_select('FPList', func_dir, 'rp_24.txt');

        %------------------- define the glm

        glm = struct();        
        glm.consess = struct([]); 
        glm.write_residuals = 1;
        display(glm)

        %------------------- run the glm
       
        glm_run(func_imgs, glm, rp_txt, [glm_dir f sub_id])
        

    catch ME

        warning('Subject %s failed: %s', sub_id, ME.message);
        continue

    end

end