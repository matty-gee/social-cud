%------------------------------------------------------------------------
% Runs Least-Squares Separate (LSS) GLMs on unsmoothed fMRIPrep BOLD images.
%
% DECISION-ONLY VERSION:
% - Only Decision trials are estimated as single-trial target regressors.
% - For each Decision trial k, design includes:
%     (1) TargetDec (trial k)
%     (2) OtherNarr (all Narrative trials)
%     (3) OtherDec (all other Decision trials)
%   + Friston-24 motion regressors
%   + explicit fMRIPrep brain mask
%   + single t-contrast on regressor 1: [1 0 0]
%
% OUTPUTS (per subject):
%   decision_trials_beta.nii.gz  (4D; one volume per decision trial, in trial order)
%   decision_trials_t.nii.gz     (4D; one volume per decision trial, in trial order)
%
% Robustness additions:
% - Per-subject try/catch: any failure -> write _FAILED.txt, cleanup, continue.
% - Per-trial try/catch: fail fast for subject (keeps output definition consistent).
% - Preflight check: nuisance rows must match #scans (truncate/pad if needed).
%
% Resume additions (this edit):
% - If intermediate 3D maps exist in _tmp_lss_decision/trialmaps_3d, reuse them.
% - Skip per-trial GLM if both cached beta/t 3D maps exist and look valid.
% - Do not wipe tmp_root at subject start when resume enabled.
% - Optionally keep cache on failure to allow resuming across runs.
%------------------------------------------------------------------------

clear; clc;

% Initialize SPM once (helps keep jobman state sane across failures)
try
    spm('defaults','FMRI');
    spm_jobman('initcfg');
catch
    % don't hard-fail here; subject-level will fail later if SPM missing
    warning('SPM init failed. Ensure SPM is on MATLAB path.');
end

model_name = 'lss';  % keep foldering the same unless you want a new model_name

% ----------------------- RESUME CONTROLS -----------------------
resume_from_existing_3d    = true;   % reuse existing 3D trial maps if present
keep_3d_cache_on_failure   = true;   % keep tmp_root on subject failure (so next run can resume)
keep_3d_after_success      = false;  % keep 3D trial maps even after final 4D is created
% --------------------------------------------------------------

f = filesep;
main_dir     = '/Users/matty_gee/Desktop/SocialDysfunction';
code_dir     = fullfile(main_dir, 'code', 'fmri_glms');
helpers_dir  = fullfile(code_dir, 'helpers');
preprc_dir   = fullfile(main_dir, 'data', 'preprocessed');
base_glm_dir = fullfile(main_dir, 'analyses');
glm_root_dir = fullfile(base_glm_dir, model_name, 'glms');

cd(code_dir);
addpath(helpers_dir);

% Optional: run-level log file
run_stamp = datestr(now,'yyyymmdd_HHMMSS');
if ~exist(glm_root_dir,'dir'), mkdir(glm_root_dir); end
run_log   = fullfile(glm_root_dir, sprintf('run_lss_decisions_%s.log', run_stamp));
fid = safe_fopen(run_log);
cobj = onCleanup(@() safe_fclose(fid));
log_line(fid, '=== run_lss_decisions started %s ===', datestr(now));

% onsets and narrative durations are same for everyone
timing = readtable('timing.xlsx');
timing = sortrows(timing, 'onset', 'ascend');

% loop over all subjects
sub_dirs = dir(fullfile(preprc_dir, 'fmriprep', 'sub*'));
sub_dirs = sub_dirs(~endsWith(string({sub_dirs.name}), ".html", "IgnoreCase", true));

for s = 3:numel(sub_dirs)

    sub_id   = sub_dirs(s).name;
    func_dir = fullfile(preprc_dir, 'fmriprep', sub_id, 'func');
    fprintf('Subject %d/%d: %s\n', s, numel(sub_dirs), sub_id);
    log_line(fid, '--- %s (index %d/%d) ---', sub_id, s, numel(sub_dirs));

    % Variables used in catch cleanup (init to safe values)
    tmp_root     = '';
    sub_out_dir  = '';
    beta4d_nii   = '';
    t4d_nii      = '';
    beta4d_gz    = '';
    t4d_gz       = '';

    try
        if ~isfolder(func_dir)
            warning('%s: func directory missing: %s', sub_id, func_dir);
            log_line(fid, '[%s] func directory missing -> skip', sub_id);
            continue
        end

        sub_out_dir = fullfile(glm_root_dir, sub_id);
        if ~exist(sub_out_dir,'dir'), mkdir(sub_out_dir); end

        % final 4D outputs
        beta4d_nii = fullfile(sub_out_dir, 'decision_trials_beta.nii');
        t4d_nii    = fullfile(sub_out_dir, 'decision_trials_t.nii');
        beta4d_gz  = [beta4d_nii '.gz'];
        t4d_gz     = [t4d_nii '.gz'];

        % Treat subject as DONE if final 4D outputs exist as .nii.gz OR .nii
        beta_final_exists = isfile(beta4d_gz) || isfile(beta4d_nii);
        t_final_exists    = isfile(t4d_gz)    || isfile(t4d_nii);

        if beta_final_exists && t_final_exists

            % If only uncompressed outputs exist, gzip them now (no re-running GLMs)
            if isfile(beta4d_nii) && ~isfile(beta4d_gz)
                try
                    gzip_and_delete(beta4d_nii);
                catch MEgz
                    warning('[%s] gzip failed for beta4d (keeping .nii): %s', sub_id, MEgz.message);
                    log_line(fid, '[%s] gzip failed for beta4d: %s', sub_id, MEgz.message);
                end
            end
            if isfile(t4d_nii) && ~isfile(t4d_gz)
                try
                    gzip_and_delete(t4d_nii);
                catch MEgz
                    warning('[%s] gzip failed for t4d (keeping .nii): %s', sub_id, MEgz.message);
                    log_line(fid, '[%s] gzip failed for t4d: %s', sub_id, MEgz.message);
                end
            end

            fprintf('Skipping (final 4D exists): %s\n', sub_id);
            log_line(fid, '[%s] Skipping: final 4D exists', sub_id);
            continue
        end

        % If partial final outputs exist (either beta or t, as .nii or .nii.gz), delete them and rebuild
        if xor(beta_final_exists, t_final_exists)
            warning('%s: Partial final outputs exist; deleting and rebuilding (will still reuse any 3D cache).', sub_id);
            log_line(fid, '[%s] Partial final 4D exists; deleting and rebuilding.', sub_id);

            safe_delete(beta4d_gz);  safe_delete(t4d_gz);
            safe_delete(beta4d_nii); safe_delete(t4d_nii);
        end

        % Ensure we rebuild cleanly
        safe_delete(beta4d_nii);
        safe_delete(t4d_nii);
        safe_delete(beta4d_gz);
        safe_delete(t4d_gz);

        % functional images
        func_base = [sub_id '_task-socialnav_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii'];
        func_imgs = ensure_nii_and_select(func_dir, func_base);

        % brain mask
        mask_base = [sub_id '_task-socialnav_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii'];
        mask_img  = ensure_nii_and_select(func_dir, mask_base);

        % motion regressors
        conf_tsv = fullfile(func_dir, [sub_id '_task-socialnav_desc-confounds_timeseries.tsv']);
        R = friston24_from_fmriprep(conf_tsv);

        % ---------------- Preflight: regressors must match scans
        nScans = numel(func_imgs);
        if size(R,1) ~= nScans
            warning('%s: Confounds rows (%d) != #scans (%d). Truncating/padding to match.', ...
                sub_id, size(R,1), nScans);
            log_line(fid, '[%s] Confounds rows (%d) != #scans (%d). Trunc/pad to match.', ...
                sub_id, size(R,1), nScans);
            R = force_nrows(R, nScans);
        end

        nuisance_txt = fullfile(func_dir, [sub_id '_task-socialnav_rp-friston24.txt']);
        writematrix(R, nuisance_txt, 'Delimiter',' ');

        % create a trial object
        trials = build_trial_table(timing, preprc_dir, sub_id);

        % identify trial types
        nT = height(trials);
        is_narr = string(trials.trial_type) == "Narrative";
        is_dec  = string(trials.trial_type) == "Decision";

        dec_idx = find(is_dec);
        nDec = numel(dec_idx);

        if nDec == 0
            warning('%s: No Decision trials found. Skipping subject.', sub_id);
            log_line(fid, '[%s] No Decision trials -> skip', sub_id);
            continue
        end

        % temp root + intermediate 3D outputs (resume-capable)
        tmp_root   = fullfile(sub_out_dir, '_tmp_lss_decision');
        tmp_3d_dir = fullfile(tmp_root, 'trialmaps_3d');

        if resume_from_existing_3d
            if ~exist(tmp_root,'dir'),   mkdir(tmp_root);   end
            if ~exist(tmp_3d_dir,'dir'), mkdir(tmp_3d_dir); end
        else
            % wipe and rebuild from scratch
            safe_rmdir(tmp_root);
            mkdir(tmp_root);
            mkdir(tmp_3d_dir);
        end

        beta_files = cell(nDec,1);
        t_files    = cell(nDec,1);

        % Pre-register expected 3D outputs; if both exist and look valid, mark as complete.
        % If only one exists (partial), delete both so we rerun that trial cleanly.
        for ii = 1:nDec
            beta_dst = fullfile(tmp_3d_dir, sprintf('decision_%04d_beta.nii', ii));
            t_dst    = fullfile(tmp_3d_dir, sprintf('decision_%04d_t.nii',    ii));

            if is_good_file(beta_dst) && is_good_file(t_dst)
                beta_files{ii} = beta_dst;
                t_files{ii}    = t_dst;
            elseif xor(isfile(beta_dst), isfile(t_dst))
                safe_delete(beta_dst);
                safe_delete(t_dst);
            end
        end

        % run GLM only for each DECISION trial
        for ii = 1:nDec

            k = dec_idx(ii);
            t = trials(k,:);

            trial_num  = double(t.trial_num);
            dn = nan;
            if ismember('decision_num', trials.Properties.VariableNames)
                dn = double(t.decision_num);
            end

            % Tag used only for temp organization / sanity
            if isfinite(dn)
                tag = sprintf('trial_%03d_decision_%03d', trial_num, dn);
            else
                tag = sprintf('trial_%03d_decision', trial_num);
            end

            % intermediate 3D files
            beta_dst = fullfile(tmp_3d_dir, sprintf('decision_%04d_beta.nii', ii));
            t_dst    = fullfile(tmp_3d_dir, sprintf('decision_%04d_t.nii',    ii));

            % If cached outputs exist, skip re-estimation
            if resume_from_existing_3d && is_good_file(beta_dst) && is_good_file(t_dst)
                fprintf('[%s] Decision %d/%d | FOUND existing 3D maps -> skipping (%s)\n', sub_id, ii, nDec, tag);
                log_line(fid, '[%s] Decision %d/%d (%s) existing 3D maps -> skip', sub_id, ii, nDec, tag);
                beta_files{ii} = beta_dst;
                t_files{ii}    = t_dst;
                continue
            end

            % Otherwise ensure we don't keep partials
            safe_delete(beta_dst);
            safe_delete(t_dst);

            fprintf('[%s] Decision %d/%d | building target regressor\n', sub_id, ii, nDec);
            log_line(fid, '[%s] Decision %d/%d (%s) RUN', sub_id, ii, nDec, tag);

            trial_glm_dir = fullfile(tmp_root, tag);
            safe_rmdir(trial_glm_dir);                 % ensure clean rerun
            if ~exist(trial_glm_dir,'dir'), mkdir(trial_glm_dir); end

            % ---- Trial-level try/catch to throw a clean error that subject-level catches
            try
                % target regressor (Decision trial k)
                cond_target = make_cond('TargetDec', t.onset, t.duration);

                % nuisance regressors: all narratives; all other decisions
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
                glm_cfg.consess = make_tcon(sprintf('%s_TargetDec', tag), [1 0 0]);

                % run GLM
                glm_run(func_imgs, glm_cfg, nuisance_txt, trial_glm_dir);

                % verify outputs exist BEFORE copy
                beta_src = fullfile(trial_glm_dir, 'beta_0001.nii');
                t_src    = fullfile(trial_glm_dir, 'spmT_0001.nii');

                if ~isfile(beta_src) || ~isfile(t_src)
                    error('Expected outputs missing after glm_run (beta_0001/spmT_0001).');
                end

                % copy the files out
                copy_one(beta_src, beta_dst);
                copy_one(t_src,    t_dst);

                % verify cached outputs look valid
                if ~is_good_file(beta_dst) || ~is_good_file(t_dst)
                    error('Copied 3D outputs failed sanity check (size/header).');
                end

                beta_files{ii} = beta_dst;
                t_files{ii}    = t_dst;

            catch ME_trial
                % Clean up the per-trial directory (best-effort)
                safe_rmdir(trial_glm_dir);

                % Delete any partial cached outputs for this trial
                safe_delete(beta_dst);
                safe_delete(t_dst);

                % Throw a clearer error upward so the SUBJECT try/catch catches it.
                ME2 = MException('run_lss_decisions:TrialFailed', ...
                    '%s: Trial failed (%s) at decision %d/%d. Root error: %s', ...
                    sub_id, tag, ii, nDec, ME_trial.message);
                ME2 = addCause(ME2, ME_trial);
                throw(ME2);
            end

            % cleanup per-trial SPM directory (success case)
            safe_rmdir(trial_glm_dir);

            if mod(ii,25) == 0
                clear batch matlabbatch glm_cfg cond_target cond_other_narr cond_other_dec;
                try spm_jobman('initcfg'); catch, end
            end
        end

        % Sanity: ensure all expected 3D maps exist before merge
        assert(all(cellfun(@isfile, beta_files)), '[%s] Missing one or more beta 3D maps before merge.', sub_id);
        assert(all(cellfun(@isfile, t_files)),    '[%s] Missing one or more t 3D maps before merge.',    sub_id);

        % MERGE TO 4D
        fprintf('[%s] Merging %d decision-trial beta maps into 4D: %s\n', sub_id, nDec, beta4d_nii);
        merge_3d_to_4d(beta_files, beta4d_nii);

        fprintf('[%s] Merging %d decision-trial t-maps into 4D: %s\n', sub_id, nDec, t4d_nii);
        merge_3d_to_4d(t_files, t4d_nii);

        % DELETE INTERMEDIATE 3D (optional)
        if ~keep_3d_after_success
            fprintf('[%s] Deleting intermediate 3D trial maps...\n', sub_id);
            for i = 1:numel(beta_files), safe_delete(beta_files{i}); end
            for i = 1:numel(t_files),    safe_delete(t_files{i});    end

            % remove temp root
            safe_rmdir(tmp_root);
        else
            fprintf('[%s] Keeping intermediate 3D trial maps (keep_3d_after_success=true).\n', sub_id);
            log_line(fid, '[%s] Keeping intermediate 3D trial maps.', sub_id);
        end

        % GZIP FINAL 4D NIFTIS + DELETE UNZIPPED
        fprintf('[%s] Gzipping final 4D NIfTIs and deleting uncompressed .nii...\n', sub_id);
        try
            gzip_and_delete(beta4d_nii);
        catch MEgz
            warning('[%s] gzip failed for beta4d (keeping .nii): %s', sub_id, MEgz.message);
            log_line(fid, '[%s] gzip failed for beta4d: %s', sub_id, MEgz.message);
        end

        try
            gzip_and_delete(t4d_nii);
        catch MEgz
            warning('[%s] gzip failed for t4d (keeping .nii): %s', sub_id, MEgz.message);
            log_line(fid, '[%s] gzip failed for t4d: %s', sub_id, MEgz.message);
        end

        fprintf('[%s] DONE. Outputs:\n  %s\n  %s\n', sub_id, beta4d_gz, t4d_gz);
        log_line(fid, '[%s] DONE.', sub_id);

    catch ME_sub
        % SUBJECT-LEVEL FAIL HANDLER:
        % - write a _FAILED.txt with full stack
        % - best-effort cleanup temp + partial outputs
        % - continue to next subject

        warning('[%s] SUBJECT FAILED. Skipping to next subject.\n%s', ...
            sub_id, getReport(ME_sub,'extended','hyperlinks','off'));
        log_line(fid, '[%s] SUBJECT FAILED: %s', sub_id, ME_sub.message);

        % Write failure report
        if ~isempty(sub_out_dir)
            failed_txt = fullfile(sub_out_dir, '_FAILED.txt');
            safe_write_text(failed_txt, sprintf('%s\n\n%s', datestr(now), getReport(ME_sub,'extended','hyperlinks','off')));
        end

        % Cleanup temp:
        % If resuming is enabled and requested, KEEP tmp_root so existing 3D maps can be reused next run.
        if ~(resume_from_existing_3d && keep_3d_cache_on_failure)
            safe_rmdir(tmp_root);
        else
            log_line(fid, '[%s] Keeping tmp_root for resume: %s', sub_id, tmp_root);
        end

        % Cleanup partial final outputs (best-effort; avoids confusing partials)
        safe_delete(beta4d_gz);  safe_delete(t4d_gz);
        safe_delete(beta4d_nii); safe_delete(t4d_nii);

        % (Re)init SPM jobman after a failure can help prevent cascading issues
        try spm_jobman('initcfg'); catch, end

        continue
    end

end

log_line(fid, '=== run_lss_decisions finished %s ===', datestr(now));

% ----------------------------- helper functions

function imgs = ensure_nii_and_select(func_dir, base_nii, delete_gz)
% ensure_nii_and_select  Ensure an unzipped .nii exists (gunzip if needed),
% optionally delete the .nii.gz after successful decompression, then return
% SPM-selected full path(s) to that .nii.
%
% delete_gz (optional): true/false, default true.
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

function R2 = force_nrows(R, nRowsTarget)
% force_nrows  Truncate or zero-pad a matrix to nRowsTarget rows.
    n = size(R,1);
    if n == nRowsTarget
        R2 = R;
    elseif n > nRowsTarget
        R2 = R(1:nRowsTarget, :);
    else
        R2 = [R; zeros(nRowsTarget-n, size(R,2))];
    end
end

function trials = build_trial_table(timing, preprc_dir, sub_id)
% build_trial_table  Merge timing (Narrative/Decision) with subject behavior.
% Returns a table with per-trial onset + duration for modeling:
%   - Narrative: duration from timing.xlsx
%   - Decision: duration = reaction_time if responded, else 0

    % timing: keep only Narrative + Decision
    timing = timing(:, :); % copy
    timing.trial_type = string(timing.trial_type);
    timing = timing(ismember(timing.trial_type, ["Narrative","Decision"]), :);

    % enforce numeric fields (robust to mixed types from Excel)
    timing.trial_num    = double(timing.trial_num);
    timing.onset        = double(timing.onset);
    timing.duration     = double(timing.duration);

    if ismember('decision_num', timing.Properties.VariableNames)
        timing.decision_num = double(string(timing.decision_num));
    else
        timing.decision_num = nan(height(timing),1);
    end

    % initialize model duration: narrative duration by default
    timing.duration_model = timing.duration;

    % behavior
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

    % RT or 0 if no response
    rt = beh.reaction_time;
    rt(~beh.responded) = 0;

    % merge decision durations onto timing
    is_dec = timing.trial_type == "Decision";
    if any(is_dec)
        [tf, loc] = ismember(timing.decision_num, beh.decision_num);
        timing.duration_model(is_dec) = rt(loc(is_dec));

        if any(is_dec & ~tf)
            bad = timing(is_dec & ~tf, {'trial_num','decision_num','onset'});
            error('%s: Unmatched decision_num(s) between timing and behavior. Example rows:\n%s', ...
                sub_id, evalc('disp(bad(1:min(5,height(bad)),:))'));
        end
    end

    % output
    trials = timing;
    trials.duration = trials.duration_model;
    trials.duration_model = [];

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

function merge_3d_to_4d(files, out_nii)
% merge_3d_to_4d  Merge a list of 3D NIfTIs into a single 4D NIfTI using SPM.
    if isempty(files)
        error('merge_3d_to_4d: empty file list for %s', out_nii);
    end
    for i = 1:numel(files)
        if ~isfile(files{i})
            error('merge_3d_to_4d: missing input file: %s', files{i});
        end
    end

    if isfile(out_nii), delete(out_nii); end

    if exist('spm_file_merge','file') == 2
        spm_file_merge(char(files), out_nii, 0);
    else
        % fallback: SPM utility "cat"
        matlabbatch = [];
        matlabbatch{1}.spm.util.cat.vols  = files(:);
        matlabbatch{1}.spm.util.cat.name  = out_nii;
        matlabbatch{1}.spm.util.cat.dtype = 0;
        matlabbatch{1}.spm.util.cat.RT    = 0;
        spm_jobman('run', matlabbatch);
    end

    if ~isfile(out_nii)
        error('merge_3d_to_4d failed to create output: %s', out_nii);
    end

    % verify volume count
    V = spm_vol(out_nii);
    if numel(V) ~= numel(files)
        error('4D merge mismatch for %s: expected %d vols, found %d', out_nii, numel(files), numel(V));
    end
end

function gz_path = gzip_and_delete(nii_path)
% gzip_and_delete  gzip a file to *.gz and delete the original *.nii
    if ~isfile(nii_path)
        error('gzip_and_delete: missing file: %s', nii_path);
    end

    gz_path = [nii_path '.gz'];
    if isfile(gz_path)
        delete(gz_path); % avoid silently keeping an old/partial .gz
    end

    gzip(nii_path);

    if ~isfile(gz_path)
        error('gzip_and_delete: gzip failed to create: %s', gz_path);
    end

    d = dir(gz_path);
    if isempty(d) || d.bytes < 1024
        error('gzip_and_delete: gzipped file looks too small (%d bytes): %s', d.bytes, gz_path);
    end

    delete(nii_path);
end

function tf = is_good_file(p)
% is_good_file  Basic sanity check for cached outputs.
% - exists
% - non-trivial size
% - (if SPM available) readable header via spm_vol
    tf = false;
    if isempty(p) || ~isfile(p), return; end

    d = dir(p);
    if isempty(d) || d.bytes < 1024
        return
    end

    if exist('spm_vol','file') == 2
        try
            V = spm_vol(p);
            tf = ~isempty(V);
        catch
            tf = false;
        end
    else
        tf = true;
    end
end

% ----------------------------- robustness helpers

function safe_delete(p)
    if isempty(p), return; end
    try
        if isfile(p), delete(p); end
    catch
        % swallow
    end
end

function safe_rmdir(p)
    if isempty(p), return; end
    try
        if exist(p,'dir'), rmdir(p,'s'); end
    catch
        % swallow
    end
end

function fid = safe_fopen(p)
    fid = [];
    try
        [d,~,~] = fileparts(p);
        if ~isempty(d) && ~exist(d,'dir'), mkdir(d); end
        fid = fopen(p,'a');
        if fid < 0, fid = []; end
    catch
        fid = [];
    end
end

function safe_fclose(fid)
    try
        if ~isempty(fid) && fid > 0
            fclose(fid);
        end
    catch
        % swallow
    end
end

function log_line(fid, varargin)
    try
        msg = sprintf(varargin{:});
    catch
        msg = '';
    end
    if ~isempty(fid) && fid > 0
        try
            fprintf(fid, '[%s] %s\n', datestr(now,'yyyy-mm-dd HH:MM:SS'), msg);
        catch
        end
    end
end

function safe_write_text(path, txt)
    try
        fid = fopen(path,'w');
        if fid < 0, return; end
        fprintf(fid, '%s\n', txt);
        fclose(fid);
    catch
        % swallow
    end
end
