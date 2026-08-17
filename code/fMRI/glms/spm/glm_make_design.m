function glm = glm_make_design(model_name, timing, behavior, out_dir, verbose, varargin)

    % Thin wrapper: get full condition spec from model, add default contrasts, save.
    if ~exist(out_dir,'dir'), mkdir(out_dir); end
    if nargin < 5, verbose = 0; end
    
    % ------------------------ helpers

    function cond = standardize_conditions(cond_in)
        cond = cond_in(:)';
        % drop obviously empty conditions early
        keep = true(1, numel(cond));
        for k = 1:numel(cond)
            % defaults
            if ~isfield(cond(k),'orth'), cond(k).orth = 0; end
            if ~isfield(cond(k),'tmod'), cond(k).tmod = 0; end
            if ~isfield(cond(k),'pmod') || isempty(cond(k).pmod)
                cond(k).pmod = struct('name',{},'param',{},'poly',{});
            end
            % shape checks
            if isempty(cond(k).onset) || isempty(cond(k).duration)
                keep(k) = false; continue
            end
            if numel(cond(k).onset) ~= numel(cond(k).duration)
                error('onset/duration length mismatch in condition %s.', cond(k).name);
            end
            if any(~isfinite(cond(k).onset)) || any(~isfinite(cond(k).duration))
                error('NaN/Inf in onsets or durations for condition %s.', cond(k).name);
            end
            if any(cond(k).duration < 0)
                error('Negative durations in condition %s.', cond(k).name);
            end
            % pmods
            pm = cond(k).pmod;
            for j = 1:numel(pm)
                for fld = ["name","param","poly"]
                    if ~isfield(pm(j), fld), error('pmod.%s missing for condition %s.', fld, cond(k).name); end
                end
                if numel(pm(j).param) ~= numel(cond(k).onset)
                    error('pmod %s length ≠ onsets for condition %s.', pm(j).name, cond(k).name);
                end
                if isempty(pm(j).poly), pm(j).poly = 1; end
            end
            % ensure within-condition onsets are ascending (warn only)
            if any(diff(cond(k).onset) < 0)
                warning('Onsets not ascending in condition %s.', cond(k).name);
            end
        end
        cond = cond(keep);
        if isempty(cond)
            error('No non-empty conditions returned by model.');
        end
        % unique names
        nm = {cond.name};
        if numel(unique(nm)) ~= numel(nm)
            error('Duplicate condition names detected: %s', strjoin(nm, ', '));
        end
    end
    
    function [names, counts] = enumerate_regressors(cond)
        % Build the regressor list in SPM's order: base, then each pmod expanded by poly.
        names  = {};
        counts = zeros(1, numel(cond));
        for k = 1:numel(cond)
            base = {cond(k).name};
            pm_names = {};
            for j = 1:numel(cond(k).pmod)
                pj = cond(k).pmod(j);
                pm_names = [pm_names, arrayfun(@(d) sprintf('%s*%s^%d', cond(k).name, pj.name, d), ...
                                               1:pj.poly, 'UniformOutput', false)]; %#ok<AGROW>
            end
            names = [names, base, pm_names]; %#ok<AGROW>
            p = [cond(k).pmod.poly];         % handles empty → []
            counts(k) = 1 + sum(p(:));       % base + sum(poly)
        end
    end
    
    function consess = make_unit_tcons(counts, reg_names)
        % One +1-vs-baseline t-contrast per regressor column.
        n_cols = sum(counts);
        consess = cell(1, numel(reg_names));
        for r = 1:numel(reg_names)
            w = zeros(1, n_cols); w(r) = 1;
            consess{r}.tcon = struct('name', [reg_names{r} ' +'], 'weights', w);
        end
    end
    
    function T = summarize_conditions(cond)
        % Produce a table with per-condition counts and duration stats.
        n = numel(cond);
        names = strings(n,1);
        n_onsets = zeros(n,1);
        d_min = zeros(n,1);
        d_med = zeros(n,1);
        d_max = zeros(n,1);
        n_pmods = zeros(n,1);
        for k = 1:n
            names(k)   = string(cond(k).name);
            n_onsets(k)= numel(cond(k).onset);
            d_min(k)   = min(cond(k).duration);
            d_med(k)   = median(cond(k).duration);
            d_max(k)   = max(cond(k).duration);
            n_pmods(k) = numel(cond(k).pmod);
        end
        T = table(names, n_onsets, d_min, d_med, d_max, n_pmods, ...
                  'VariableNames', {'condition','n_onsets','dur_min','dur_med','dur_max','n_pmods'});
    end
    
    %------------------------ 1) get the model specification

    fh = str2func(['models.' model_name]);
    if ~isempty(varargin)
        [cond_spec, opts] = fh(timing, behavior, varargin{:});  % e.g., target_idx for LSS
    else
        [cond_spec, opts] = fh(timing, behavior);               % legacy models
    end
    if ~isfield(opts,'write_residuals'), opts.write_residuals = 0; end
    
    %------------------------ 2) define conditions + checks + summaries

    glm.cond = standardize_conditions(cond_spec);
    Csum = summarize_conditions(glm.cond);
    [reg_names, reg_counts] = enumerate_regressors(glm.cond);
    consess = make_unit_tcons(reg_counts, reg_names);
    
    % (Optional) append any model-provided extra contrasts
    if isfield(opts,'extra_consess') && ~isempty(opts.extra_consess)
        consess = [consess, opts.extra_consess];
    end
    glm.consess = consess;
    
    % ------------------------ Housekeeping
    
    glm.write_residuals = opts.write_residuals;
    
    if verbose
        fprintf('Design for model "%s": %d conditions -> %d regressors, %d contrasts.\n', ...
                model_name, numel(glm.cond), sum(reg_counts), numel(glm.consess));
        disp(Csum);
    end
    
    writetable(cell2table(reg_names(:),'VariableNames',{'regressor'}), ...
               fullfile(out_dir,'design_regressors.csv'));
    writetable(Csum, fullfile(out_dir,'design_conditions.csv'));
    tcon_names = cellfun(@(c) c.tcon.name, glm.consess, 'UniformOutput', false);
    writetable(cell2table(tcon_names(:),'VariableNames',{'tcontrast'}), ...
               fullfile(out_dir,'design_tcontrasts.csv'));
    
    save(fullfile(out_dir,'design.mat'), '-struct', 'glm');
end
