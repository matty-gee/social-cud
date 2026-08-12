%------------------------------------------------------------------------
% Runs Least-Squares Separate (LSS) GLMs on unsmoothed fMRIPrep BOLD images.
%
% Modeling choices:
% - Two event types are modeled: Narrative and Decision (Image slides excluded).
% - Narrative events use boxcar durations from the shared timing file (timing.xlsx).
% - Decision events use boxcar durations equal to the subject's reaction time (RT)
% from the behavioral file; trials without a response are modeled with duration = 0
% (impulse), preserving a consistent design across trials.
% - LSS is implemented as one GLM per trial. For each trial k, the design includes:
% (1) a single-trial target regressor (Narrative_k or Decision_k),
% (2) a nuisance regressor aggregating all other Narrative trials,
% (3) a nuisance regressor aggregating all other Decision trials.
% Narrative and Decision "other-trial" regressors are kept separate because they
% differ systematically in duration and timing, improving control of shared variance.
% - Motion nuisance regressors are Friston-24 extracted from fMRIPrep confounds.
% - An explicit fMRIPrep brain mask is applied during model specification.
% - For each trial, the target beta (beta_0001) and its corresponding t-map (spmT_0001)
% are saved with standardized filenames; temporary SPM outputs are deleted.
%------------------------------------------------------------------------

clear; clc;

model_name = 'lss';


f = filesep;
main_dir     = '/Users/matty_gee/Desktop/SocialDysfunction';
code_dir     = fullfile(main_dir, 'code', 'fmri_glms');
helpers_dir  = fullfile(code_dir, 'helpers');
preprc_dir   = fullfile(main_dir, 'data', 'preprocessed');
base_glm_dir = fullfile(main_dir, 'analyses');
glm_dir      = fullfile(base_glm_dir, model_name, 'glms');
cd(code_dir);
addpath(helpers_dir);

% onsets and narrative durations are same for everyon
timing = readtable('timing.xlsx');
timing = sortrows(timing, 'onset', 'ascend');

% loop over all subjects
sub_dirs = dir(fullfile(preprc_dir, 'fmriprep', 'sub*'));
sub_dirs = sub_dirs(~endsWith(string({sub_dirs.name}), ".html", "IgnoreCase", true));

for s = 1 % :numel(sub_dirs)

    sub_id   = sub_dirs(s).name;
    func_dir = fullfile(preprc_dir, 'fmriprep', sub_id, 'func');
    fprintf('Subject %d/%d: %s\n', s, numel(sub_dirs), sub_id);
    if ~isfolder(func_dir)
        warning('%s: func directory missing: %s', sub_id, func_dir);
        continue
    end
    sub_out_dir = fullfile(glm_dir, sub_id);
    if ~exist(sub_out_dir,'dir'), mkdir(sub_out_dir); end

    % functional images
    func_base = [sub_id '_task-socialnav_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii'];
    func_imgs = ensure_nii_and_select(func_dir, func_base);
    
    % brain mask
    mask_base = [sub_id '_task-socialnav_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii'];
    mask_img  = ensure_nii_and_select(func_dir, mask_base);

    % motion regressors
    conf_tsv = fullfile(func_dir, [sub_id '_task-socialnav_desc-confounds_timeseries.tsv']);
    R = friston24_from_fmriprep(conf_tsv);
    nuisance_txt = fullfile(func_dir, [sub_id '_task-socialnav_rp-friston24.txt']);
    writematrix(R, nuisance_txt, 'Delimiter',' ');

    % create a trial object
    trials = build_trial_table(timing, preprc_dir, sub_id);  

    % output dirs
    beta_out = fullfile(sub_out_dir, 'lss_betas');
    t_out    = fullfile(sub_out_dir, 'lss_tmaps');
    if ~exist(beta_out,'dir'), mkdir(beta_out); end
    if ~exist(t_out,'dir'),    mkdir(t_out);    end
        tmp_root = fullfile(sub_out_dir, '_tmp_lss');
    if ~exist(tmp_root, 'dir'), mkdir(tmp_root); end
    
    % run glm for each trial
    nT = height(trials);
    is_narr = string(trials.trial_type) == "Narrative";
    is_dec  = string(trials.trial_type) == "Decision";
    for k = 1:nT

        t = trials(k,:);
        trial_num  = double(t.trial_num);
        trial_type = lower(string(t.trial_type));   % "narrative" or "decision"
        tag = sprintf('trial_%03d_%s', trial_num, trial_type);
        if trial_type == "narrative" && ismember('narrative_num', trials.Properties.VariableNames)

            nn = double(t.narrative_num);
            if isfinite(nn)
                tag = sprintf('trial_%03d_narrative_%03d', trial_num, nn);
            end

            cond_target = make_cond('TargetNarr', t.onset, t.duration);
            con_name    = 'TargetNarr';

        elseif trial_type == "decision" && ismember('decision_num', trials.Properties.VariableNames)

            dn = double(t.decision_num);
            if isfinite(dn)
                tag = sprintf('trial_%03d_decision_%03d', trial_num, dn);
            end

            cond_target = make_cond('TargetDec', t.onset, t.duration); %  RT or 0 
            con_name    = 'TargetDec';

        end
        fprintf('tag=%s | beta=%s | t=%s\n', tag, [tag '_beta.nii'], [tag '_t.nii']);
        glm_dir = fullfile(tmp_root, tag);
        if ~exist(glm_dir,'dir'), mkdir(glm_dir); end
    
        other_narr_idx  = find(is_narr & ((1:nT)' ~= k));
        other_dec_idx   = find(is_dec  & ((1:nT)' ~= k));
        cond_other_narr = make_cond('OtherNarr', trials.onset(other_narr_idx), trials.duration(other_narr_idx));
        cond_other_dec  = make_cond('OtherDec',  trials.onset(other_dec_idx),  trials.duration(other_dec_idx));

        % glm configuration
        glm_cfg = struct();
        glm_cfg.cond = [cond_target, cond_other_narr, cond_other_dec];
        glm_cfg.mthresh = -Inf;
        glm_cfg.mask = mask_img;
        glm_cfg.write_residuals = 0;
        glm_cfg.consess = make_tcon(sprintf('%s_%s', tag, con_name), [1 0 0]); % single t-contrast on regressor 1
    
        % check if file exists
        beta_dst = fullfile(beta_out, [tag '_beta.nii']);
        t_dst    = fullfile(t_out,    [tag '_t.nii']);
        
        if isfile(beta_dst) && isfile(t_dst)
            fprintf('Skipping (exists): %s\n', tag);
            continue
        end

        % run GLM
        glm_run(func_imgs, glm_cfg, nuisance_txt, glm_dir);

        % copy the files & cleanup
        copy_one(fullfile(glm_dir, 'beta_0001.nii'), beta_dst);
        copy_one(fullfile(glm_dir, 'spmT_0001.nii'), t_dst);
        try
            rmdir(glm_dir, 's');
        catch ME
            warning('Could not delete temp dir %s (%s).', glm_dir, ME.message);
        end    
        if mod(k,25) == 0
            clear batch matlabbatch glm_cfg cond_target cond_other_narr cond_other_dec;
            try spm_jobman('initcfg'); catch, end
        end
        
    end
    
    % remove the temp root if empty
    try
        if exist(tmp_root,'dir')
            d = dir(tmp_root);
            if numel(d) <= 2, rmdir(tmp_root); end
        end
    catch
    end

end

% ----------------------------- helper functions

function imgs = ensure_nii_and_select(func_dir, base_nii, delete_gz)
% ensure_nii_and_select  Ensure an unzipped .nii exists (gunzip if needed),
% optionally delete the .nii.gz after successful decompression, then return
% SPM-selected full path(s) to that .nii.
%
% delete_gz (optional): true/false, default false.

    if nargin < 3, delete_gz = true; end

    nii_path = fullfile(func_dir, base_nii);
    gz_path  = [nii_path '.gz'];

    if ~exist(nii_path, 'file')
        if ~exist(gz_path, 'file')
            error('Missing file: %s (and %s)', nii_path, gz_path);
        end

        gunzip(gz_path);

        % sanity check: .nii created and non-empty
        if ~exist(nii_path, 'file')
            error('gunzip reported success but .nii not found: %s', nii_path);
        end
        d = dir(nii_path);
        if isempty(d) || d.bytes < 1024
            error('Decompressed .nii looks too small (%d bytes): %s', d.bytes, nii_path);
        end

        % delete .gz only after successful verification
        if delete_gz
            delete(gz_path);
        end
    end

    imgs = cellstr(spm_select('ExtFPList', func_dir, ['^' base_nii '$']));
end

function R = friston24_from_fmriprep(tsv_path)
% Return N×24 Friston-24 motion regressors from fMRIPrep confounds TSV

    T = readtable(tsv_path, 'FileType','text', 'Delimiter','\t');

    base = {'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'};
    deriv = strcat(base, '_derivative1');
    base2 = strcat(base, '_power2');
    deriv2 = strcat(deriv, '_power2');

    cols = [base, deriv, base2, deriv2];

    % keep only columns that exist (robust to version differences)
    have = intersect(cols, T.Properties.VariableNames, 'stable');
    if numel(have) ~= 24
        error('Expected 24 Friston columns, found %d', numel(have));
    end

    R = T{:, have};
    R(~isfinite(R)) = 0;   % fMRIPrep often has NaNs in first derivative row
end

function trials = build_trial_table(timing, preprc_dir, sub_id)
% build_trial_table  Merge timing (Narrative/Decision) with subject behavior.
% Returns a table with per-trial onset + duration for modeling:
%   - Narrative: duration from timing.xlsx
%   - Decision: duration = reaction_time if responded, else 0

    % -------------------- timing: keep only Narrative + Decision --------------------
    timing = timing(:, :); % copy
    timing.trial_type = string(timing.trial_type);
    timing = timing(ismember(timing.trial_type, ["Narrative","Decision"]), :);

    % enforce numeric fields (robust to mixed types from Excel)
    timing.trial_num    = double(timing.trial_num);
    timing.onset        = double(timing.onset);
    timing.duration     = double(timing.duration);

    if ismember('decision_num', timing.Properties.VariableNames)
        % decision_num may be 'n/a' for narratives; coerce safely
        timing.decision_num = double(string(timing.decision_num));
    else
        timing.decision_num = nan(height(timing),1);
    end

    % initialize model duration: narrative duration by default
    timing.duration_model = timing.duration;

    % -------------------- behavior (your exact loading pattern) --------------------
    beh_file = fullfile(preprc_dir, 'behavior', [sub_id '.xlsx']);
    if ~isfile(beh_file)
        error('%s: behavior file missing: %s', sub_id, beh_file);
    end
    beh = readtable(beh_file);
    beh = sortrows(beh, 'decision_num', 'ascend');

    beh.decision_num  = double(beh.decision_num);
    beh.reaction_time = double(beh.reaction_time);

    % responded can be logical or TRUE/FALSE strings; normalize to logical
    if ismember('responded', beh.Properties.VariableNames)
        if iscell(beh.responded) || isstring(beh.responded)
            beh.responded = strcmpi(string(beh.responded), "TRUE");
        else
            beh.responded = logical(beh.responded);
        end
    else
        beh.responded = true(height(beh),1); % fallback if absent
    end

    % Option A: RT or 0 if no response
    rt = beh.reaction_time;
    rt(~beh.responded) = 0;

    % -------------------- merge decision durations onto timing --------------------
    is_dec = timing.trial_type == "Decision";
    if any(is_dec)
        [tf, loc] = ismember(timing.decision_num, beh.decision_num);
        timing.duration_model(is_dec) = rt(loc(is_dec));  % loc is valid where tf is true

        % fail loudly if any decision trials didn't match behavior
        if any(is_dec & ~tf)
            bad = timing(is_dec & ~tf, {'trial_num','decision_num','onset'});
            error('%s: Unmatched decision_num(s) between timing and behavior. Example rows:\n%s', ...
                sub_id, evalc('disp(bad(1:min(5,height(bad)),:))'));
        end
    end

    % -------------------- output: final per-trial table to loop over --------------------
    trials = timing;
    trials.duration = trials.duration_model;
    trials.duration_model = [];

    % sanity: no missing onsets/durations
    assert(all(isfinite(trials.onset)),    'Non-finite onsets in trials table.');
    assert(all(isfinite(trials.duration)), 'Non-finite durations in trials table.');
end

function c = make_cond(name, onsets, durs)
    c.name     = name;
    c.onset    = onsets(:)';   % row vector
    c.duration = durs(:)';     % row vector
    c.tmod     = 0;
    c.pmod     = struct('name', {}, 'param', {}, 'poly', {});
    c.orth     = 1;
end

function consess = make_tcon(name, w)
    consess = {struct('tcon', struct( ...
        'name',    name, ...
        'weights', w, ...
        'sessrep', 'none'))};
end

function copy_one(src, dst)
    if ~isfile(src)
        error('Expected file missing: %s', src);
    end
    [ok,msg] = copyfile(src, dst);
    if ~ok, error('copyfile failed: %s -> %s\n%s', src, dst, msg); end
end