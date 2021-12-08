%%% arpad attila voros
%% training data for latent space projection
clc; clear;

% load in variables
datapath = "data/";
datafolder = "data0";
load(strcat(datapath, datafolder, "/", datafolder, ".mat"), 'datafolder', 'datapath', 'dims', ...
    'lref', 'seen_labels', 'unseen_labels', 'unseen_uniq_labels');
X_unseen_feat_ = importdata(strcat(datapath, datafolder, "/classify1/unseendata_f_classify1.mat"));
X_seen_feat = importdata(strcat(datapath, datafolder, "/classify1/seendata_f_classify1.mat"));
% X_weight_init = X_seen_feat'*X_seen_feat;

% number of latent variables for latent space projection
num_latent_vars = 256;

%% iterate SAE to get latent projection weights used in ZSL
% one-hot encode seen labels
Y_train = label_matrix(seen_labels')';

% intialize weights from feature to encoded space
covar_strength = 50;
W_feat = (X_weight_init + covar_strength*eye(size(X_seen_feat, 2)))^(-1)*X_seen_feat'*Y_train;
X_seen = X_seen_feat * W_feat;

% get features of seen data, initialize latent projection matrix
S_train_lbl = randn(length(lref), num_latent_vars);
S_seen = zeros(size(X_seen_feat, 1), num_latent_vars);
for t_datapoint = 1:size(X_seen_feat, 1)
    S_seen(t_datapoint, :) = S_train_lbl(seen_labels(t_datapoint) + 1, :);
end

% iterate SAE to minimize weights which go to/from latent space
lambda = 1;
W_latent = SAE(X_seen', S_seen', lambda)';
num_iter = 25;
for iter = 1:num_iter
    W_latent = SAE(X_seen', (X_seen * W_latent)', lambda)';
    fprintf("%.1f%%\n", 100 * iter / num_iter);
end

%% randomly sample unseen data to act as semantic baseline
X_unseen_baseline = zeros(length(unseen_uniq_labels), size(X_unseen_feat_, 2));
X_unseen_feat = X_unseen_feat_;
% one of each untrained feature
xidx = 1;
for lbl_actual = unseen_uniq_labels'
    lbl_indices = find(unseen_labels == lbl_actual);
    representative_lbl_idx = lbl_indices(randi(length(lbl_indices)));
    X_unseen_baseline(xidx, :) = X_unseen_feat(representative_lbl_idx, :);
    X_unseen_feat(representative_lbl_idx, :) = [];
    xidx = xidx + 1;
end
X_unseen = X_unseen_feat * W_feat;

%% get semantic representation of seen (trained) data + semantic baseline
X_seen = X_seen_feat * W_feat;
S_seen = X_seen * W_latent;
seen_uniq_labels = unique(seen_labels);
num_seen_labels = length(seen_uniq_labels);
S_seen_baseline = zeros(num_seen_labels, num_latent_vars);
for lbl_idx = 1:length(seen_uniq_labels)
    S_seen_baseline(lbl_idx, :) = mean(S_seen(seen_labels == seen_uniq_labels(lbl_idx), :));
%     S_seen_baseline(lbl_idx, :) = abs(exp(mean(log(S_seen(seen_labels == seen_uniq_labels(lbl_idx), :))))) ...
%         .* sign(mean(S_seen(seen_labels == seen_uniq_labels(lbl_idx), :)));
end
all_uniq_labels = [seen_uniq_labels; unseen_uniq_labels];
all_labels = [seen_labels; unseen_labels];

%% get semantic baseline of unseen (test) data + all seen+unseen data
% get semantic baseline using first instances of randomly sampled data
S_unseen_baseline = X_unseen_baseline * W_feat * W_latent;
S_omega_baseline = [S_seen_baseline; S_unseen_baseline];
S_unseen = X_unseen * W_latent;
S_all = [S_seen; S_unseen];

% top # accuracy
num_hits = 1;

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

%% latent space -> feature space of untrained data
X_unseen_feat_predict = (W_feat' \ (S_unseen * W_latent')')';

%% load untrained data + decoder for displaying
load(strcat(datapath, datafolder, "/", datafolder, ".mat"), 'seen_data');
load(strcat(datapath, datafolder, "/", datafolder, ".mat"), 'unseen_data');
load(strcat(datapath, datafolder, "/classify1/decode_net_classify1_TEST_BS.mat"));
% feature_size = [7, 7, 32];
reps = 1;
% feature_size = [reps, reps, 26];

%% display random unseen result
num_iter = 1;
for iter = 1:num_iter
    rand_idx = randi(length(unseen_labels));
%     lbl_actual = unseen_labels(rand_idx);
%     lbl_prdict = Y_hits(rand_idx, 1);
%     disp_title = sprintf("Actual: %s\nGuess: %s", ...
%         get_refs_string(get_refs(lref, lbl_actual)), get_refs_string(get_refs(lref, lbl_prdict)));
    
    figure(1);
    imdisp(reshape(unseen_data(rand_idx, :), dims, dims));
%     title(disp_title);
%     saveas(gcf, strcat(datapath, datafolder, "/classify1/p", string(iter), ".png"));
    % title(strcat("Input data === ", disp_title));

%     Y_from_latent = predict(decode_net, reshape(X_unseen_feat_predict(rand_idx, :), feature_size(1), feature_size(2), feature_size(3), 1));
%     Y_from_latent = predict(decode_net, reshape(repmat(X_unseen(rand_idx, :), 1, reps*reps), feature_size(1), feature_size(2), feature_size(3), 1));
    Y_from_latent = predict(decode_net, reshape(repmat(X_unseen(rand_idx, :)*W_feat', 1, reps*reps), feature_size(1), feature_size(2), feature_size(3), 1));
%     Y_from_latent = predict(decode_net, ...
%         reshape(predict(decode_net_lf, ...
%             reshape(repmat(X_unseen(rand_idx, :), 1, reps*reps), ...
%             feature_size(1), feature_size(2), feature_size(3), 1)), 7, 7, 32));
    figure(2);
    imdisp(reshape(Y_from_latent, dims, dims));
%     title(strcat("Output predicted from latent space === ", disp_title));
end

%% display random seen result
num_iter = 1;
for iter = 1:num_iter
    rand_idx = randi(length(seen_labels));
%     lbl_actual = seen_labels(rand_idx);
%     lbl_prdict = Y_hits(rand_idx, 1);
%     disp_title = sprintf("Actual: %s\nGuess: %s", ...
%         get_refs_string(get_refs(lref, lbl_actual)), get_refs_string(get_refs(lref, lbl_prdict)));
    
    figure(1);
    imdisp(reshape(seen_data(rand_idx, :), dims, dims));
%     title(disp_title);
%     saveas(gcf, strcat(datapath, datafolder, "/classify1/p", string(iter), ".png"));
    % title(strcat("Input data === ", disp_title));

%     Y_from_latent = predict(decode_net, reshape(X_seen_feat(rand_idx, :), feature_size(1), feature_size(2), feature_size(3), 1));
%     Y_from_latent = predict(decode_net, reshape(repmat(X_seen(rand_idx, :), 1, reps*reps), feature_size(1), feature_size(2), feature_size(3), 1));
    Y_from_latent = predict(decode_net, reshape(repmat(X_seen(rand_idx, :)*W_feat', 1, reps*reps), feature_size(1), feature_size(2), feature_size(3), 1));
%     Y_from_latent = predict(decode_net, ...
%         reshape(predict(decode_net_lf, ...
%             reshape(repmat(X_seen(rand_idx, :), 1, reps*reps), ...
%             feature_size(1), feature_size(2), feature_size(3), 1)), 7, 7, 32));
    figure(2);
    imdisp(reshape(Y_from_latent, dims, dims));
%     title(strcat("Output predicted from latent space === ", disp_title));
end