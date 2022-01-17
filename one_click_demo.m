% ONE CLICK DEMO
% INSTRUCTIONS:
% - SELECT DATAFOLDER (LINE 13 / 24)
% - SELECT TOP ACCURACY (LINE 58)
% - RUN ALL

% mainly taken from eval_model.m, where training SAE and other stuff
% excluded. trained autoencoders used in report are instead loaded in

% LOAD DATA FROM PATH, CAN CHANGE
% LOAD DATA FROM PATH, CAN CHANGE
% LOAD DATA FROM PATH, CAN CHANGE
% datafolder = "data0";
datafolder = "data1";
% LOAD DATA FROM PATH, CAN CHANGE
% LOAD DATA FROM PATH, CAN CHANGE
% LOAD DATA FROM PATH, CAN CHANGE

% load in variables
datapath = "data/";
load(strcat(datapath, datafolder, "/", datafolder, ".mat"), 'datafolder', 'datapath', 'dims', ...
    'lref', 'seen_labels', 'unseen_labels', 'unseen_uniq_labels');
X_unseen_feat_ = importdata(strcat(datapath, datafolder, "/classify1/unseendata_f_classify1.mat"));
X_seen_feat = importdata(strcat(datapath, datafolder, "/classify1/seendata_f_classify1.mat"));

% number of latent variables for latent space projection
num_latent_vars = 256;

% select right model from report
if datafolder == "data0"
    load(strcat(datapath, datafolder, "/classify1/data_838.mat"));
elseif datafolder == "data1"
    load(strcat(datapath, datafolder, "/classify1/data_538.mat"));
end

% calculate semantic baselines
X_seen = X_seen_feat * W_feat;
S_seen = X_seen * W_latent;
seen_uniq_labels = unique(seen_labels);
num_seen_labels = length(seen_uniq_labels);
S_seen_baseline = zeros(num_seen_labels, num_latent_vars);
for lbl_idx = 1:length(seen_uniq_labels)
    S_seen_baseline(lbl_idx, :) = mean(S_seen(seen_labels == seen_uniq_labels(lbl_idx), :));
end
all_uniq_labels = [seen_uniq_labels; unseen_uniq_labels];
all_labels = [seen_labels; unseen_labels];

S_unseen_baseline = X_unseen_baseline * W_feat * W_latent;
S_omega_baseline = [S_seen_baseline; S_unseen_baseline];
S_unseen = X_unseen * W_latent;
S_all = [S_seen; S_unseen];

fprintf("%d (unseen classes) / %d (total classes)\n", length(unseen_uniq_labels), length(unseen_uniq_labels) + num_seen_labels);

% TOP # ACCURACY, CAN EDIT
% TOP # ACCURACY, CAN EDIT
% TOP # ACCURACY, CAN EDIT
num_hits_arr = [1, 3];
% TOP # ACCURACY, CAN EDIT
% TOP # ACCURACY, CAN EDIT
% TOP # ACCURACY, CAN EDIT

for num_hits = num_hits_arr
    warning off
    % z-score of distance to determine closest datapoint in latent space
    % unseen data mapped to ==> unseen semantic space
    dist = zscore(1 - pdist2((S_unseen), (S_unseen_baseline), 'cosine'));
    Y_hits = zeros(size(dist, 1), num_hits);
    for i = 1:size(dist, 1)
        [~, I] = sort(dist(i, :), 'descend');
        Y_hits(i, :) = unseen_uniq_labels(I(1:num_hits));
    end
    
    n = 0;
    for i = 1:size(dist, 1)
        if ismember(unseen_labels(i), Y_hits(i, :))
            n = n + 1;
        end
    end
    zsl_accuracy = n / size(dist, 1);
    fprintf('[%d] ZSL accuracy [unseen ==> unseen]: %.1f%%\n', num_hits, zsl_accuracy * 100);
    
    % z-score of distance to determine closest datapoint in latent space
    % unseen data mapped to ==> unseen+seen semantic space
    dist = zscore(1 - pdist2((S_unseen), (S_omega_baseline), 'cosine'));
    Y_hits = zeros(size(dist, 1), num_hits);
    for i = 1:size(dist, 1)
        [~, I] = sort(dist(i, :), 'descend');
        Y_hits(i, :) = all_uniq_labels(I(1:num_hits));
    end
    
    n = 0;
    for i = 1:size(dist, 1)
        if ismember(unseen_labels(i), Y_hits(i, :))
            n = n + 1;
        end
    end
    zsl_accuracy = n / size(dist, 1);
    fprintf('[%d] ZSL accuracy [unseen ==> all]: %.1f%%\n', num_hits, zsl_accuracy * 100);
    
    % z-score of distance to determine closest datapoint in latent space
    % seen data mapped to ==> seen semantic space
    dist = zscore(1 - pdist2((S_seen), (S_seen_baseline), 'cosine'));
    Y_hits = zeros(size(dist, 1), num_hits);
    for i = 1:size(dist, 1)
        [~, I] = sort(dist(i, :), 'descend');
        Y_hits(i, :) = seen_uniq_labels(I(1:num_hits));
    end
    
    n = 0;
    for i = 1:size(dist, 1)
        if ismember(seen_labels(i), Y_hits(i, :))
            n = n + 1;
        end
    end
    zsl_accuracy = n / size(dist, 1);
    fprintf('[%d] ZSL accuracy [seen ==> seen]: %.1f%%\n', num_hits, zsl_accuracy * 100);
    
    % z-score of distance to determine closest datapoint in latent space
    % seen data mapped to ==> unseen+seen semantic space
    dist = zscore(1 - pdist2((S_seen), (S_omega_baseline), 'cosine'));
    Y_hits = zeros(size(dist, 1), num_hits);
    for i = 1:size(dist, 1)
        [~, I] = sort(dist(i, :), 'descend');
        Y_hits(i, :) = all_uniq_labels(I(1:num_hits));
    end
    
    n = 0;
    for i = 1:size(dist, 1)
        if ismember(seen_labels(i), Y_hits(i, :))
            n = n + 1;
        end
    end
    zsl_accuracy = n / size(dist, 1);
    fprintf('[%d] ZSL accuracy [seen ==> all]: %.1f%%\n', num_hits, zsl_accuracy * 100);
    
    % z-score of distance to determine closest datapoint in latent space
    % all data mapped to ==> unseen+seen semantic space
    dist = zscore(1 - pdist2((S_all), (S_omega_baseline), 'cosine'));
    Y_hits = zeros(size(dist, 1), num_hits);
    for i = 1:size(dist, 1)
        [~, I] = sort(dist(i, :), 'descend');
        Y_hits(i, :) = all_uniq_labels(I(1:num_hits));
    end
    
    n = 0;
    for i = 1:size(dist, 1)
        if ismember(all_labels(i), Y_hits(i, :))
            n = n + 1;
        end
    end
    zsl_accuracy = n / size(dist, 1);
    fprintf('[%d] ZSL accuracy [all ==> all]: %.1f%%\n', num_hits, zsl_accuracy * 100);

end