function spm_write_vol_4D(V, Y)
    for i = 1:numel(V)
        Vi = V(i);
        Vi.n = [i 1];  % set frame index
        spm_write_vol(Vi, Y(:,:,:,i));
    end
end