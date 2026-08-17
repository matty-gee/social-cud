%------------------------------------------------------------------------
% Runs GLMs on UNSMOOTHED images
% Run prepare_data.m first
%------------------------------------------------------------------------

clear; clc;

model_name = 'lsa_fmriprep';

f = filesep;
cd '/Users/matty_gee/Desktop/SocialDysfunction/code/fmri_glms'
preprc_dir   = '/Users/matty_gee/Desktop/SocialDysfunction/data/preprocessed';
base_glm_dir = '/Users/matty_gee/Desktop/SocialDysfunction/analyses';
glm_dir      = [base_glm_dir f model_name f 'glms'];

% timing is the same for everyone
timing = readtable('timing.xlsx'); 
timing = sortrows(timing, 'onset', 'ascend');

% loop over all subjects
sub_dirs = dir(fullfile([preprc_dir f 'fmri'],'sub*'));
for s = 1 : numel(sub_dirs)

    sub_id   = sub_dirs(s).name; 
    func_dir = [preprc_dir f 'fmri' f sub_id f 'func'];
    out_dir  = [glm_dir f sub_id];
    fprintf('Processing %d/%d: %s\n', s, numel(sub_dirs), sub_id);

    try

        if ~isfolder(func_dir)
            warning('%s: func directory missing: %s', sub_id, func_dir);
            continue
        end
    
        %------------------- load subject data

        beh  = readtable([preprc_dir f 'behavior' f sub_id '.xlsx']);
        beh  = sortrows(beh, 'decision_num', 'ascend');
        imgs = cellstr(spm_select('ExtFPList', func_dir, 'func.nii'));
        rp   = spm_select('FPList', func_dir, 'rp.txt');

        if isempty(imgs), warning('%s: no func.nii', sub_id); continue; end
        if isempty(rp),   warning('%s: no rp.txt',  sub_id); continue; end

        %------------------- define & run the glm

        glm = glm_make_design(model_name, timing, beh, out_dir, 0);
        glm_run(imgs, glm, rp, out_dir)

    catch ME

        warning('%s failed: %s', sub_id, ME.message);
        continue

    end

end