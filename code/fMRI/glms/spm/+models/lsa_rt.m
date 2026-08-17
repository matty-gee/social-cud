function [cond, opts] = lsa_rt(timing, behavior)

    %------------------------------------------------------------------------
    % least squares all for decision trials
    % reaction time HRF modeling
    %------------------------------------------------------------------------

    %------------------- Basic checks

    opts     = struct();
    is_dec   = strcmp(timing.slide_type,'Decision');
    is_narr  = strcmp(timing.slide_type,'Narrative');
    assert(any(is_narr), 'Expected a Narrative block in timing, but none was found.');

    dec_tbl  = timing(is_dec,:);
    n_trials = height(dec_tbl);
    assert(n_trials == height(behavior), 'Decision timing and behavior must align.');
    
    %------------------- Preallocate

    n_cond   = 1 + n_trials;
    empty_c  = struct('name','','onset',[],'duration',[],'orth',0,'tmod',0, ...
                      'pmod',struct('name',{},'param',{},'poly',{}));
    cond     = repmat(empty_c, 1, n_cond);
    
    %------------------- Narrative trials in single regressor

    cond(1).name     = 'narrative';
    cond(1).onset    = timing.onset(is_narr);
    cond(1).duration = timing.duration(is_narr);
    cond(1).orth     = 0;

    %------------------- Decision trials each get their own regressors

    for t = 1 : n_trials
        cond(1+t).name     = sprintf('decision_%02d', behavior.decision_num(t));
        cond(1+t).onset    = dec_tbl.onset(t);
        cond(1+t).duration = behavior.reaction_time(t);   
        cond(1+t).orth     = 0;                    
    end

end
