function [data, labels] = get_bindata(paths)
% get_bindata -  ARPAD ATTILA VOROS
%   pulls from .mat files, returns data (logical) and labels
%   INPUTS:     paths - string array, each to .mat path + filename
%                       .mat files have ONE 2d array stored
%                       all .mat files SAME SIZE AND FORMAT
%                       all .mat files SORTED
%                       .mat FILE FORMAT:
%                           arr1_data =   importdata([path(1), "_d.mat"]);
%                           arr1_labels = importdata([path(1), "_l.mat"]);
%                           labels are valued 0 -> n
%                           data is BINARY
%   OUTPUTS:    data - data (assumed logical from .mat) appended
%               labels - labels (assumed 0 -> n for each .mat, therefore based
%               on order of paths, labels incremented for appendage
%               e.g., paths(1) labels =  0  ->  n
%                     paths(2) labels = n+1 -> n+m

num_paths = length(paths);
data = logical([]);
labels = [];
labels_offset = 0;
for pidx = 1:num_paths
    % pull data
    dataset =  importdata(strcat(paths(pidx), "_d.mat"));
    labelset = importdata(strcat(paths(pidx), "_l.mat"));
    % append to output
    data = [data; logical(dataset)];
    labels = [labels; labelset + labels_offset];
    % increment label offset
    labels_offset = labels_offset + max(labelset) + 1;
end

end