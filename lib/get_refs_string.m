function disp_str = get_refs_string(refs)
% 
disp_str = string(char(refs(1)));
for sim_idx = 2:length(refs)
    disp_str = strcat(disp_str, " or ", string(char(refs(sim_idx))));
end
end