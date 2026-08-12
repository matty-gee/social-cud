%------------------------------------------------------------------------
% Helper to run some initial prep before fMRI modeling
% 1) smooth the functional images
% 2) generate text files with 24 motion parameters
%------------------------------------------------------------------------

clear all;

f = filesep;
cd '/Users/matty_gee/Desktop/SocialTrajectories/code/fmri_glms'
preprc_dir = '/Users/matty_gee/Desktop/SocialTrajectories/data/preprocessed';

% loop over all subjects
subdirs = dir(fullfile([preprc_dir f 'fmri'],'sub*'));
for s = 1 : numel(subdirs)

    sub_id = subdirs(s).name; 
    fprintf('Now processing %d/%d: %s\n', s, numel(subdirs), sub_id);

    try
        
        % smooth data
        func_fname = 'func_smoothed6.nii';
        func_dir   = [preprc_dir f 'fmri' f sub_id f 'func'];
        if ~isfile([func_dir f func_fname]) 
            img_smooth(spm_select('ExtFPList', func_dir, 'func.nii'), 6)
        end
        
        % add rp 24 file
        make_rp24(func_dir)

    catch ME

        warning('Subject %s failed: %s', sub_id, ME.message);
        continue

    end

end