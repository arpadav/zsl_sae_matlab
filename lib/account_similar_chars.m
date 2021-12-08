function [labels, lref] = account_similar_chars(labels, sim_data, ref_idx, lrefmat_base)
% e.g. :
% sim_data = {["0", "O"]; ...
%             ["1", "I", "L"]; ...
%             ["S", "5"]};
% ref_idx = [1, 1, 2]; means 0, 1, and 5 are references

% store reference into cell array
num_labels = size(lrefmat_base, 1);
lref = cell(num_labels, 1);
for lbl_idx = 1:num_labels
    lref{lbl_idx} = lrefmat_base(lbl_idx, :);
    for char_idx = 1:length(sim_data)
        chars = double(reshape(char(sim_data{char_idx}), [1 length(sim_data{char_idx})]));
        if max(chars == lrefmat_base(lbl_idx, 2))
            lref{lbl_idx} = [lref{lbl_idx}(1), chars];
        end
    end
end

for char_idx = 1:length(sim_data)
    % loop through similar characters and replace labels
    ref_ascii = double(char(sim_data{char_idx}(ref_idx(char_idx))));
    ref_label = lrefmat_base(lrefmat_base(:, 2) == ref_ascii, 1);
    for sim_idx = 1:length(sim_data{char_idx})
        % for all characters which are NOT reference character, replace labels
        if sim_idx ~= ref_idx(char_idx)
            sim_ascii = double(char(sim_data{char_idx}(sim_idx)));
            sim_label = lrefmat_base(lrefmat_base(:, 2) == sim_ascii, 1);
            labels(labels == sim_label) = ref_label;
        end
    end
end

end