%------------------------------------------------------------------------
% Runs Least-Squares All (LSA) GLM on unsmoothed fMRIPrep BOLD images.
%
% DECISION-ONLY VERSION:
% - Only Decision trials are included in the LSA design (one regressor per decision trial).
%
% Adds:
% (1) subject-level skip if merged outputs already exist
% (2) merges per-trial betas and t-maps into 4D NIfTIs:
%     decision_trials_beta.nii
%     decision_trials_t.nii
%------------------------------------------------------------------------

clear; clc;

model_name = 'lsa_decision';

f = filesep;
main_dir     = '/Users/matty_gee/Desktop/SocialDysfunction';
code_dir     = fullfile(main_dir, 'code', 'fmri_glms');
helpers_dir  = fullfile(code_dir, 'helpers');
preprc_dir   = fullfile(main_dir, 'data', 'preprocessed');
base_glm_dir = fullfile(main_dir, 'analyses');
glm_dir      = fullfile(base_glm_dir, model_name, 'glms');
cd(code_dir);
addpath(helpers_dir);

% onsets and narrative durations are the same for everyone
try
    timing = readtable('timing.xlsx');
    timing = sortrows(timing, 'onset', 'ascend');
catch ME
    fprintf(2, 'ERROR: Failed to read timing.xlsx: %s\n', ME.message);
    fprintf(2, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
    return
end

% loop over all subjects
sub_dirs = dir(fullfile(preprc_dir, 'fmriprep', 'sub*'));
sub_dirs = sub_dirs(~endsWith(string({sub_dirs.name}), ".html", "IgnoreCase", true));

for s = 1 :numel(sub_dirs)

    sub_id   = sub_dirs(s).name;
    func_dir = fullfile(preprc_dir, 'fmriprep', sub_id, 'func');
    fprintf('Subject %d/%d: %s\n', s, numel(sub_dirs), sub_id);
    if ~isfolder(func_dir)
        warning('%s: func directory missing: %s', sub_id, func_dir);
        continue
    end

    % Reset per-subject temp-dir handle so catch() never references prior subject
    run_dir = '';

    try
        sub_out_dir = fullfile(glm_dir, sub_id);
        if ~exist(sub_out_dir,'dir'), mkdir(sub_out_dir); end

        % Skip subject if merged outputs already exist (DECISION ONLY)
        out_dec_beta = fullfile(sub_out_dir, 'decision_trials_beta.nii');
        out_dec_t    = fullfile(sub_out_dir, 'decision_trials_t.nii');

        if isfile(out_dec_beta) && isfile(out_dec_t)
            fprintf('Skipping %s (decision merged outputs exist)\n', sub_id);
            continue
        end

        % functional images
        func_base = [sub_id '_task-socialnav_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii'];
        func_imgs = ensure_nii_and_select(func_dir, func_base, '4d');
        if isempty(func_imgs) || isempty(func_imgs{1})
            error('%s: func_imgs is empty. Check filename/path matching. func_dir=%s/%s', sub_id, func_dir, func_base);
        end

        % brain mask
        mask_base = [sub_id '_task-socialnav_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii'];
        mask_img  = ensure_nii_and_select(func_dir, mask_base, '3d');

        % motion regressors (Friston-24)
        conf_tsv = fullfile(func_dir, [sub_id '_task-socialnav_desc-confounds_timeseries.tsv']);
        R = friston24_from_fmriprep(conf_tsv);
        nuisance_txt = fullfile(func_dir, [sub_id '_task-socialnav_rp-friston24.txt']);
        writematrix(R, nuisance_txt, 'Delimiter',' ');

        % build per-trial table (Narrative + Decision), then KEEP DECISION ONLY
        trials = build_trial_table(timing, preprc_dir, sub_id);
        trials = trials(string(trials.trial_type) == "Decision", :);
        if isempty(trials)
            error('%s: No Decision trials found after filtering.', sub_id);
        end

        % output dirs (per-trial outputs)
        beta_out = fullfile(sub_out_dir, 'lsa_betas');
        t_out    = fullfile(sub_out_dir, 'lsa_tmaps');
        if ~exist(beta_out,'dir'), mkdir(beta_out); end
        if ~exist(t_out,'dir'),    mkdir(t_out);    end

        % ---------------------------------------------------------------------
        % Build LSA design: one regressor per DECISION trial
        % ---------------------------------------------------------------------

        nT = height(trials);

        cond = struct('name', {}, 'onset', {}, 'duration', {}, ...
                      'tmod', {}, 'pmod', {}, 'orth', {});
        tags = cell(nT,1);

        for k = 1:nT
            t = trials(k,:);

            trial_num  = double(t.trial_num);
            trial_type = lower(string(t.trial_type)); % should be "decision"
            tag = sprintf('trial_%03d_%s', trial_num, trial_type);

            if trial_type == "decision" && ismember('decision_num', trials.Properties.VariableNames)
                dn = safe_num(t.decision_num);
                if isfinite(dn)
                    tag = sprintf('trial_%03d_decision_%03d', trial_num, dn);
                end
            end

            tags{k} = tag;

            % SPM condition names must start with a letter
            cond_name = ['T_' tag];
            cond_name = regexprep(cond_name, '[^A-Za-z0-9_]', '_');

            cond(k) = make_cond(cond_name, t.onset, t.duration);
        end

        % ---------------------------------------------------------------------
        % One t-contrast per trial regressor (identity matrix)
        % ---------------------------------------------------------------------

        consess = cell(1, nT);
        for k = 1:nT
            w = zeros(1, nT);
            w(k) = 1;
            consess{k} = struct('tcon', struct( ...
                'name',    [tags{k} '_T'], ...
                'weights', w, ...
                'sessrep', 'none'));
        end

        % ---------------------------------------------------------------------
        % Run single LSA GLM
        % ---------------------------------------------------------------------

        run_dir = fullfile(sub_out_dir, '_tmp_lsa');
        if exist(run_dir,'dir'), rmdir(run_dir,'s'); end
        mkdir(run_dir);

        glm_cfg = struct();
        glm_cfg.cond = cond;
        glm_cfg.mthresh = -Inf;
        glm_cfg.mask = mask_img;
        glm_cfg.write_residuals = 0;
        glm_cfg.consess = consess;

        glm_run(func_imgs, glm_cfg, nuisance_txt, run_dir);

        % ---------------------------------------------------------------------
        % Copy per-trial betas and t-maps
        % ---------------------------------------------------------------------

        for k = 1:nT
            beta_src = fullfile(run_dir, sprintf('beta_%04d.nii', k));
            t_src    = fullfile(run_dir, sprintf('spmT_%04d.nii', k));

            beta_dst = fullfile(beta_out, [tags{k} '_beta.nii']);
            t_dst    = fullfile(t_out,    [tags{k} '_t.nii']);

            copy_one(beta_src, beta_dst);
            copy_one(t_src,    t_dst);
        end

        % ---------------------------------------------------------------------
        % Merge per-trial betas and t-maps into 4D stacks (DECISION ONLY)
        % ---------------------------------------------------------------------

        beta_paths = cell(nT,1);
        t_paths    = cell(nT,1);
        for k = 1:nT
            beta_paths{k} = fullfile(beta_out, [tags{k} '_beta.nii']);
            t_paths{k}    = fullfile(t_out,    [tags{k} '_t.nii']);
        end

        merge_3d_to_4d(beta_paths, out_dec_beta);
        merge_3d_to_4d(t_paths,    out_dec_t);

        % cleanup
        try
            rmdir(run_dir, 's');
        catch ME
            warning('Could not delete temp dir %s (%s).', run_dir, ME.message);
        end

    catch ME

        % Catch *any* per-subject error (loading images, glm_run, merges, file ops, etc.)
        fprintf(2, '\nERROR processing %s: %s\n', sub_id, ME.message);
        fprintf(2, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));

        % Best-effort cleanup if temp dir was created
        if ~isempty(run_dir) && isfolder(run_dir)
            try
                rmdir(run_dir, 's');
            catch ME2
                warning('Could not delete temp dir %s after failure (%s).', run_dir, ME2.message);
            end
        end

        % Move on to next subject
        continue
    end

end

% ----------------------------- helper functions

function x = safe_num(v)
    x = str2double(string(v));
    if ~isfinite(x), x = NaN; end
end

function imgs = ensure_nii_and_select(func_dir, base_nii, kind, delete_gz)
% ensure_nii_and_select
% - Ensures an unzipped .nii exists (gunzip if needed; optionally deletes .nii.gz)
% - Then returns scan strings in the *robust* format SPM wants:
%     kind = '4d'  -> {'/path/file.nii,1'; '/path/file.nii,2'; ...}
%     kind = '3d'  -> {'/path/file.nii'}
    if nargin < 3 || isempty(kind), kind = '4d'; end
    if nargin < 4, delete_gz = true; end

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

    switch lower(string(kind))
        case "3d"
            imgs = {nii_path};

        case "4d"
            V = spm_vol(nii_path);
            nvol = numel(V);
            if nvol < 2
                warning('Expected 4D image but spm_vol returned %d volume(s): %s', nvol, nii_path);
            end
            imgs = cell(nvol, 1);
            for i = 1:nvol
                imgs{i} = sprintf('%s,%d', nii_path, i);
            end

        otherwise
            error('Unknown kind="%s". Use "3d" or "4d".', kind);
    end
end

function merge_3d_to_4d(paths_3d, out_4d)
% merge_3d_to_4d  Merge a list of 3D NIfTIs into a single 4D NIfTI.
% Uses spm_file_merge. Preserves ordering of input list.
    if isempty(paths_3d)
        fprintf('Skipping merge (empty list): %s\n', out_4d);
        return
    end

    for i = 1:numel(paths_3d)
        if ~isfile(paths_3d{i})
            error('Missing image for merge: %s', paths_3d{i});
        end
    end

    if isfile(out_4d)
        delete(out_4d);
    end

    P = char(cellfun(@(p) sprintf('%s,1', p), paths_3d, 'UniformOutput', false));
    spm_file_merge(P, out_4d, 0);

    fprintf('Wrote 4D merge: %s  (n=%d)\n', out_4d, numel(paths_3d));
end

function R = friston24_from_fmriprep(tsv_path)
    T = readtable(tsv_path, 'FileType','text', 'Delimiter','\t');

    base   = {'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'};
    deriv  = strcat(base, '_derivative1');
    base2  = strcat(base, '_power2');
    deriv2 = strcat(deriv, '_power2');

    cols = [base, deriv, base2, deriv2];
    have = intersect(cols, T.Properties.VariableNames, 'stable');
    if numel(have) ~= 24
        error('Expected 24 Friston columns, found %d', numel(have));
    end

    R = T{:, have};
    R(~isfinite(R)) = 0;
end

function trials = build_trial_table(timing, preprc_dir, sub_id)
    timing = timing(:, :);
    timing.trial_type = string(timing.trial_type);
    timing = timing(ismember(timing.trial_type, ["Narrative","Decision"]), :);

    timing.trial_num = double(timing.trial_num);
    timing.onset     = double(timing.onset);
    timing.duration  = double(timing.duration);

    if ismember('decision_num', timing.Properties.VariableNames)
        timing.decision_num = double(string(timing.decision_num));
    else
        timing.decision_num = nan(height(timing),1);
    end

    timing.duration_model = timing.duration;

    beh_file = fullfile(preprc_dir, 'behavior', [sub_id '.xlsx']);
    if ~isfile(beh_file)
        error('%s: behavior file missing: %s', sub_id, beh_file);
    end
    beh = readtable(beh_file);
    beh = sortrows(beh, 'decision_num', 'ascend');

    beh.decision_num  = double(beh.decision_num);
    beh.reaction_time = double(beh.reaction_time);

    if ismember('responded', beh.Properties.VariableNames)
        if iscell(beh.responded) || isstring(beh.responded)
            beh.responded = strcmpi(string(beh.responded), "TRUE");
        else
            beh.responded = logical(beh.responded);
        end
    else
        beh.responded = true(height(beh),1);
    end

    rt = beh.reaction_time;
    rt(~beh.responded) = 0;

    is_dec = timing.trial_type == "Decision";
    if any(is_dec)
        [tf, loc] = ismember(timing.decision_num, beh.decision_num);
        timing.duration_model(is_dec) = rt(loc(is_dec));
        if any(is_dec & ~tf)
            bad = timing(is_dec & ~tf, {'trial_num','decision_num','onset'});
            error('%s: Unmatched decision_num(s) between timing and behavior.\n%s', ...
                sub_id, evalc('disp(bad(1:min(5,height(bad)),:))'));
        end
    end

    trials = timing;
    trials.duration = trials.duration_model;
    trials.duration_model = [];

    assert(all(isfinite(trials.onset)),    'Non-finite onsets in trials table.');
    assert(all(isfinite(trials.duration)), 'Non-finite durations in trials table.');
end

function c = make_cond(name, onsets, durs)
    c.name     = name;
    c.onset    = onsets(:)';
    c.duration = durs(:)';
    c.tmod     = 0;
    c.pmod     = struct('name', {}, 'param', {}, 'poly', {});
    c.orth     = 1;
end

function copy_one(src, dst)
    if ~isfile(src)
        error('Expected file missing: %s', src);
    end
    [ok,msg] = copyfile(src, dst);
    if ~ok
        error('copyfile failed: %s -> %s\n%s', src, dst, msg);
    end
end
