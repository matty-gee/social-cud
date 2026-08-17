function glm_design = glm_make_design(model_name, timing, behavior, glm_dir, verbose)

    %------------------------------------------------------------------------------
    % Make the design for a first-level GLM; also specifies some basic t-contrasts
    % 
    % 
    % Arguments
    % ---------
    % model : str
    %   specifies the model to build
    % timing : table
    %   contains onset information
    % behavior : table
    %   contains behavioral information: e.g., reaction-time, values for
    %   parametric modulators etc
    % glm_dir : str
    %   where to output
    % verbose : bool
    % 
    % 
    % Returns
    % -------
    % glm_design : struct
    %
    %
    % Some notes
    % ----------
    % expects angles in radians
    %------------------------------------------------------------------------------
    
    if ~exist(glm_dir,'dir'), mkdir(glm_dir); end
    
    %------------------------ helpers
    
    cond_struct = @(n,o,d,t) struct('name', n, 'onset', o, ...
                                    'duration', d, 'trials', t, 'orth', 1,...
                                    'pmod', struct('name', {}, 'param', {}, ...
                                    'poly', {}, 'normalized', {}));  
    event_durations = struct('onset', zeros(63,1), ...
                            'epoch', ones(63,1) * 12, ...
                            'decision', behavior.reaction_time);
    normalize = struct( ...
        'none',   @(X) X, ...
        'z',      @(X) zscore(X), ...
        'cos',    @(X) cos(X), ...
        'sin',    @(X) sin(X), ...
        'demean', @(X) X - mean(X), ...
        'zcos',   @(X) zscore(cos(X)), ...
        'zcos_d', @(X) zscore(1-cos(X)), ...
        'zsin',   @(X) zscore(sin(X)), ...
        'zabs',   @(X) zscore(abs(X)) ...
    );
    
    function C = make_conditions(n, name_prefix, eventType, pmods)
        % returns an n×3 cell array: { name, eventType, pmods }
        names = arrayfun(@(i) sprintf('%s%02d', name_prefix, i), ...
                         1:n, 'UniformOutput', false)';
        C = cell(n,3);
        C(:,1)      = names;
        C(:,2)      = repmat({eventType}, n, 1);
        C(:,3)      = repmat({pmods},    n, 1);
    end
    
    %------------------------ define model 
    
    neg_tcons = 1; % whether to output negative t-contrasts
    model = {};    % default
    switch model_name
        case {'lsa_onset', 'lsa_decision'}
            event     = extractAfter(model_name, "lsa_"); % gives 'onset' or 'decision'
            model     = make_conditions(63, "trial", event, {}); 
            neg_tcons = 0; % no need to make negative contrasts
            
        case 'angle' % only in half-space, so only need cosine
            model = {'all','decision', {'pov_angle','zcos'}};
            
        case 'distance'
            model = {'all','decision', {'pov_dist','z'}};
    
        case 'angle_delta' % assumes only magnitude of change matters
            model = {'all','decision', {'pov_angle_delta','zabs'}};
            
        case 'distance_delta'
            model = {'all','decision', {'pov_dist_delta','z'}};
            
        case 'polar'
            model = {'all','decision', {'pov_angle','zcos'; 'pov_dist','z'}};
                            
        case 'dimension'
            model = {'affil','decision',[]; 'power','decision',[] };
            
        case 'character'
            model = make_conditions(5, "character", "decision", {});
            
        case 'character_polar'
            pmods = {'pov_angle','zcos'; 'pov_dist','z'};
            model = make_conditions(5, "character", "decision", pmods);
    
        otherwise
            error("Unknown model_name '%s'", model_name);
    end
    
    % if the model is character-wise w/ pmods, we need to specify each character condition
    if strcmp(model{1}, 'character') 
        tmp = {};
        for n_char = 1:5
            tmp = [tmp; {sprintf('character0%d', n_char), model{2}, model{3}}];
        end
        model = tmp;
    end
    
    [n_conds, ~] = size(model);
    [n_pmods, ~] = size(vertcat(model{:, 3}));
    
    
    %------------------------ define conditions
    
    % define narrative condition first
    narrative = timing(strcmp(timing.slide_type, 'Narrative'), :);
    regrs = {'narrative'};
    glm_design.cond(1) = cond_struct('narrative', narrative.onset, narrative.duration, []);
    n_conds = n_conds + 1;
    
    
    % define the rest
    decisions = timing(strcmp(timing.slide_type, 'Decision'), :);
    for n_cond = 2:n_conds 
    
        [cond_trials, bold_event, cond_pmods] = model{n_cond-1, :}; % unpack the model
    
        % define event onsets
        if contains(cond_trials, 'trial') % single trial
    
            n_trial     = str2double(erase(cond_trials, 'trial'));
            cond_name   = sprintf('decision_%02d', n_trial);
            trials_incl = behavior.decision_num == n_trial;
    
        elseif strcmp(cond_trials, 'all')
    
            cond_name   = 'decisions';
            trials_incl = ones(63,1) == 1;
    
        elseif strcmp(cond_trials, 'affil') || strcmp(cond_trials, 'power')
    
            cond_name   = cond_trials;
            trials_incl = strcmp(behavior.dimension, cond_trials);
    
        elseif contains(cond_trials, 'character')
    
            n_char      = str2double(erase(cond_trials, 'character'));
            cond_name   = sprintf('character_%02d', n_char);
            trials_incl = behavior.char_role_num == n_char;
    
        end
    
        glm_design.cond(n_cond) = cond_struct(cond_name, decisions(trials_incl, :).onset, [], trials_incl);
    
        % assign durations [& optional parametric modulators] to events
        trials_incl = glm_design.cond(n_cond).trials;
    
        % duration
        glm_design.cond(n_cond).duration = event_durations.(bold_event)(trials_incl); 
        regrs = [regrs, glm_design.cond(n_cond).name];
    
        % pmods [optional]
        glm_design.cond(n_cond).orth = 0; % turn off orthogonalization
        [n_cond_pmods, ~] = size(cond_pmods);
        for n_pmod = 1:n_cond_pmods
    
            [pmod_name, normlz] = cond_pmods{n_pmod, :};
            
            if ~strcmp(normlz, 'none')
               regr_name = [glm_design.cond(n_cond).name '*' normlz '(' pmod_name ')'];
            else
               regr_name = [glm_design.cond(n_cond).name '*' pmod_name]; 
            end
            regrs = [regrs, regr_name];
            param = behavior(trials_incl, pmod_name).Variables; 
            param = normalize.(normlz)(param);
            glm_design.cond(n_cond).pmod(n_pmod) = struct('name', pmod_name, 'param', param,...
                                                          'poly', 1, 'normalized', normlz);
    
        end
    end
    
    glm_design.write_residuals = 0;
    
    n_tps   = length(vertcat(glm_design.cond.onset)); % number of timepoints
    n_regrs = length(regrs); % number of regressors
    
    %------------------------ weight contrasts
    
    % weight individual regrs against baseline
    tcons = {};    
    for n_regr = 1:n_regrs
    
        tcon_weights = zeros(1, length(regrs));
        tcon_weights(n_regr) = 1;
    
        % positive 
        tcon_name = [regrs{n_regr} '+'];
        tcons     = [tcons, tcon_name];
        glm_design.consess{length(tcons)}.tcon = struct('name', tcon_name, 'weights',  tcon_weights);
    
        % negative
        if neg_tcons
            tcon_name = [regrs{n_regr} '-'];
            tcons     = [tcons, tcon_name];
            glm_design.consess{length(tcons)}.tcon = struct('name', tcon_name, 'weights', -tcon_weights);
        end
    end
    n_tcons = length(tcons);
    
    %------------------------ do some checks
    
    check_sum = @(to_sum, exp_sum) sum(to_sum) == exp_sum;
    
    % check number of events in diff conditions (excls narrative)
    if n_conds-1 ~= 63
        if n_conds-1 == 1
            cond_ntrials_exp = 63;
        elseif n_conds-1 == 2 % assuming a power/affil breakdown
            cond_ntrials_exp = [30 30];
        elseif n_conds-1 == 5 % assuming characterwise 
            cond_ntrials_exp = [12 12 12 12 12];
        end
        cond_ntrials = sum(horzcat(glm_design.cond(2:end).trials), 1);
        if check_sum(cond_ntrials == cond_ntrials_exp, n_conds-1) == 0
            error('Number of trials per condition is off')
        end
    end
    
    % check number of regressors (incls narrative)
    if check_sum((n_pmods + n_conds), n_regrs) == 0
        error('Number of regressors is off')
    end
    
    if verbose 
        fprintf('%d timepoints\n', n_tps)
        fprintf('%d conditions with %d total param. mod.\n', n_conds, n_pmods)
        fprintf('%d regressors: %s\n', n_regrs, sprintf('%s; ', regrs{:}))
        fprintf('%d contrasts: %s\n', n_tcons, sprintf('%s; ', tcons{:}))
    end
    
    %------------------------ output stuff
    
    f = filesep;
    glm_design.model_array = model;
    save([glm_dir f 'design.mat'], '-struct', 'glm_design')
    writetable(cell2table(regrs), [glm_dir f 'design_regressors.csv'])
    writetable(cell2table(tcons), [glm_dir f 'design_tcontrasts.csv'])
    
    %------------------------ clean up
    
    glm_design.cond = rmfield(glm_design.cond, 'trials');
    for n_cond = 1:length(glm_design.cond)
        glm_design.cond(n_cond).pmod = rmfield(glm_design.cond(n_cond).pmod, 'normalized');
    end
    
    display(glm_design)

end