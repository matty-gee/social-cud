function make_rp24(rp_dir)

% MAKE_RP24  Compute Friston‐24 motion regressors from an SPM realignment file.
%   make_rp24(rpDir) looks for a file named rp*.txt in rp_dir, loads the six
%   realignment parameters, computes first‐order derivatives and squares,
%   and saves the 24‐column result as rp_24.txt in rp_dir

    % find the realignment file
    files = dir(fullfile(rp_dir,'rp.txt'));
    if isempty(files)
        error('No realignment file found in %s', rp_dir);
    elseif numel(files)>1
        warning('Multiple matches; using %s', files(1).name);
    end
    infile = fullfile(rp_dir, files(1).name);

    % load the 6 rigid‐body parameters [nScans × 6]
    rp = load(infile);  

    % compute first‐order temporal derivatives
    drp = [zeros(1,size(rp,2)); diff(rp)];

    % square originals and derivatives
    rp2  = rp .^ 2;
    drp2 = drp.^ 2;

    % concatenate into [nScans × 24]
    rp24 = [rp, drp, rp2, drp2];

    % save out as rp_24.txt
    outfile = fullfile(rp_dir,'rp_24.txt');
    save(outfile,'rp24','-ascii');

end
