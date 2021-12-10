%%% 633 project workbench - ZSL
%%% arpad attila voros
%% pull data
% paths to .mat data (see get_bindata for comments on formatting)
paths = [   "data/raw/digits", ...
            "data/raw/alphabet_uppercase"];
[data, labels] = get_bindata(paths);

% dimensions of data, 2d images from MNIST
dims = 28;

% reference labels table (for display and output)
load("data/lrefmat.mat");

%% account for similar characters
% e.g. :
% sim_data = {["0", "O"]; ...
%             ["1", "I", "L"]; ...
%             ["S", "5"]};
% ref_idx = [1, 1, 2]; means 0, 1, and 5 are references

% similar characters
sim_data = {["0", "O"]; ...
            };
% which index of characters to have as REFERENCE LABEL
% i.e., all other characters will be RELABELED to reference label
ref_idx = ones(numel(sim_data), 1);

% store reference into cell array
num_labels = size(lrefmat, 1);
lref = cell(num_labels, 1);
for lbl_idx = 1:num_labels
    lref{lbl_idx} = lrefmat(lbl_idx, :);
    for char_idx = 1:length(sim_data)
        chars = double(reshape(char(sim_data{char_idx}), [1 length(sim_data{char_idx})]));
        if max(chars == lrefmat(lbl_idx, 2))
            lref{lbl_idx} = [lref{lbl_idx}(1), chars];
        end
    end
end

for char_idx = 1:length(sim_data)
    % loop through similar characters and replace labels
    ref_ascii = double(char(sim_data{char_idx}(ref_idx(char_idx))));
    ref_label = lrefmat(lrefmat(:, 2) == ref_ascii, 1);
    for sim_idx = 1:length(sim_data{char_idx})
        % for all characters which are NOT reference character, replace labels
        if sim_idx ~= ref_idx(char_idx)
            sim_ascii = double(char(sim_data{char_idx}(sim_idx)));
            sim_label = lrefmat(lrefmat(:, 2) == sim_ascii, 1);
            labels(labels == sim_label) = ref_label;
        end
    end
end

%% randomly display data

% randomly select data point
disp_idx = randi(size(data, 1));

% display
figure(1);
imshow(reshape(data(disp_idx, :), [dims, dims]));
sim_chars = char(lref{labels(disp_idx) + 1}(2:end));
disp_str = string(sim_chars(1));
for sim_idx = 2:length(sim_chars)
    disp_str = strcat(disp_str, " or ", string(sim_chars(sim_idx)));
end
title(strcat("Unicode: ", disp_str));

%% rand sample test labels + data for training

% get unique labels
uniq_labels = unique(labels);
num_uniq_labels = length(uniq_labels);

% ratio of seen and unseen labels for ZSL
seen_ratio = 3;
num_unseen = round(num_uniq_labels / (seen_ratio + 1));
% unseen_uniq_labels = sort(datasample(uniq_labels, num_unseen, 'Replace', false));
unseen_labels_hotidx = sum(labels' == unseen_uniq_labels)';
seen_data =     data(find(~unseen_labels_hotidx), :);   %#ok<FNDSB> 
unseen_data =   data(find(unseen_labels_hotidx), :);    %#ok<FNDSB> 
seen_labels =   labels(find(~unseen_labels_hotidx), :); %#ok<FNDSB> 
unseen_labels = labels(find(unseen_labels_hotidx), :);  %#ok<FNDSB> 

%% datapath for processed data
datapath = "data/";
datafolder = "data1";
% save to datapath for training
save(strcat(datapath, datafolder, "/", datafolder, ".mat"), "dims", "lref", ...
    "unseen_uniq_labels", "seen_data", "unseen_data", "seen_labels", "unseen_labels", ...
    "datapath", "datafolder");

%% actually train network to get features
clear; clc;
load("data/data0/data0.mat");

xtrain = reshape(seen_data', dims, dims, 1, size(seen_data, 1));
ytrain = categorical(seen_labels);
seen_uniq_labels = unique(seen_labels);
num_final_layer = length(seen_uniq_labels);

%%
layers = [
    imageInputLayer([dims dims 1])
    
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(num_final_layer)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');
%%
net = trainNetwork(xtrain, ytrain, layers, options);
% save(strcat(datapath, datafolder, "/classify1/net_classify1.mat"), "net");

%% get features
layer = 'relu_3';
seen_data_features = activations(net, reshape(seen_data', dims, dims, 1, size(seen_data, 1)), layer, 'OutputAs', 'rows');
% seen_data_features = reshape(seen_data_features', 7, 7, 32, size(seen_data_features, 1));
unseen_data_features = activations(net, reshape(unseen_data', dims, dims, 1, size(unseen_data, 1)), layer, 'OutputAs', 'rows');
% unseen_data_features = reshape(unseen_data_features', 7, 7, 32, size(unseen_data_features, 1));

% % save to datapath for decoder training
% save(strcat(datapath, datafolder, "classify1/seendata_f_classify1.mat"), "seen_data_features");
% save(strcat(datapath, datafolder, "classify1/unseendata_f_classify1.mat"), "unseen_data_features");

%%
% img8 = reshape(logical(rgb2gray(imread("data/data0/eight.png"))), dims, dims, 1, 1);
% imgM = reshape(logical(rgb2gray(imread("data/data0/em.png"))), dims, dims, 1, 1);
% imgP = reshape(logical(rgb2gray(imread("data/data0/pe.png"))), dims, dims, 1, 1);
% disp(char(lref{seen_uniq_labels(double(classify(net, img8))) + 1}(2)));
% disp(char(lref{seen_uniq_labels(double(classify(net, imgM))) + 1}(2)));
% disp(char(lref{seen_uniq_labels(double(classify(net, imgP))) + 1}(2)));
% disp(char(lref{seen_uniq_labels(double(classify(net, reshape(unseen_data(7000, :), dims, dims)))) + 1}(2)));

%%
for lidx = 1:length(lref)
    if ismember(lidx, seen_uniq_labels)
        disp(char(lref{lidx}(2)));
    end
end

%%
% seen_data_features = reshape(seen_data_features', 7, 7, 32, size(seen_data_features, 1));
% seen_data = reshape(seen_data', dims, dims, 1, size(seen_data, 1));

%% suprisingly worked lol
layersGenerator = [
    imageInputLayer([7 7 32],"Name","imageinput","Normalization","none")
%     fullyConnectedLayer(prod(projectionSize))
%     functionLayer(@(X) feature2image(X,projectionSize),Formattable=true)

    transposedConv2dLayer([3 3],32,"Name","transposed-conv_1","Cropping","same")
    batchNormalizationLayer
%     reluLayer

    transposedConv2dLayer([3 3],16,"Name","transposed-conv_2","Cropping","same")
    batchNormalizationLayer
%     reluLayer

    transposedConv2dLayer([3 3],8,"Name","transposed-conv_3","Cropping","same")
    fullyConnectedLayer(784,"Name","fc")
%     tanhLayer
    sigmoidLayer
    regressionLayer];

%%
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

decode_net = trainNetwork(reshape(seen_data_features', 7, 7, 32, size(seen_data_features, 1)), ...
    double(seen_data), layersGenerator, options);

%% test decoder output
layer = 'relu_3';
imgM = logical(rgb2gray(imread("data/data0/em.png")));
imgM_feat = activations(net, reshape(imgM, dims, dims, 1, 1), layer, 'OutputAs', 'rows');
% imdisp(reshape(imgM_feat, 7*4, 7*8));
imgM_feat = reshape(imgM_feat, 7, 7, 32, 1);

layer = 'relu_3';
img_test = logical(rgb2gray(imread("data/data0/pi.png")));
img_test_feat = activations(net, reshape(img_test, dims, dims, 1, 1), layer, 'OutputAs', 'rows');
imdisp(reshape(img_test_feat, 7*4, 7*8));
img_test_feat = reshape(img_test_feat, 7, 7, 32, 1);

test = predict(decode_net, img_test_feat);
imdisp(reshape(test, dims, dims));
%% do SAE stuff

% X_trf = seen_data_features;
X_trf = importdata("data/data0/classify1/seendata_f_classify1.mat");
num_latent_vars = 256;

S_train_lbl = randn(length(lref), num_latent_vars);
S_train = zeros(size(X_trf, 1), num_latent_vars);

for t_datapoint = 1:size(X_trf, 1)
    S_train(t_datapoint, :) = S_train_lbl(seen_labels(t_datapoint) + 1, :);
end

%% continue top, just do not want to rerun
% uniq_idx = unique(seen_labels) + 1;

% X_trf = seen_data_features;
% X_trf = importdata("data/data0/classify1/seendata_f_classify1.mat");
Y     = label_matrix(seen_labels')';
W_old = (X_trf'*X_trf + 150 * eye(size(X_trf'*X_trf)))^(-1)*X_trf'*Y;
X_tr  = X_trf * W_old;
%%
% unseen_uniq_labels = unique(unseen_labels);
% X_tef = unseen_data_features;
X_tef = importdata("data/data0/classify1/unseendata_f_classify1.mat");
X_test_pos = zeros(length(unseen_uniq_labels), size(X_tef, 2));
% one of each untrained feature
xidx = 1;
for lbl = unseen_uniq_labels'
    lbl_indices = find(unseen_labels == lbl);
    representative_lbl_idx = lbl_indices(randi(length(lbl_indices)));
    X_test_pos(xidx, :) = X_tef(representative_lbl_idx, :);
    X_tef(representative_lbl_idx, :) = [];
    xidx = xidx + 1;
end

X_te  = X_tef * W_old;
%% ???? how to embed / eval single instance to get semantic representation
% S_test_pos = NormalizeFea(X_test_pos * W_old * W);
S_test_pos = X_test_pos * W_old * W;

% X_test_pos_rep = zeros(length(unseen_labels), size(X_test_pos, 2));
% for lbl_idx = 1:length(unseen_uniq_labels)
%     X_test_pos_rep(unseen_labels == unseen_uniq_labels(lbl_idx), :) = repmat(X_test_pos(lbl_idx, :), sum(unseen_labels == unseen_uniq_labels(lbl_idx), 'all'), 1);
% end
% %%
% S_test_pos = randn(length(unseen_uniq_labels), num_latent_vars);
% % ^^ THIS HOW TO DO IT, BUT EXTEND ?
% % S_test_pos = randn(length(unseen_labels), num_latent_vars);
% 
% % Y_unseen = label_matrix(unseen_labels')';
% Y_unseen_uniq = label_matrix(unseen_uniq_labels')';
% % W_old_gt = (X_test_pos_rep'*X_test_pos_rep + 150 * eye(size(X_test_pos_rep'*X_test_pos_rep)))^(-1)*X_test_pos_rep'*Y_unseen;
% W_old_gt = (X_test_pos'*X_test_pos + 150 * eye(size(X_test_pos'*X_test_pos)))^(-1)*X_test_pos'*Y_unseen_uniq;
% % X_tp  = X_test_pos_rep * W_old_gt;
% X_tp  = X_test_pos * W_old_gt;
% 
% lambda  = 1;
% % S_train    = NormalizeFea(S_train);
% W_ground_truth = SAE(X_tp', S_test_pos', lambda)';
% 
% num_iter = 20;
% for iter = 1:num_iter
%     W_ground_truth = SAE(X_tp', (X_tp*W_ground_truth)', lambda)';
%     disp(100 * iter / num_iter);
% end
% S_test_pos = X_tp * W_ground_truth;

%%
% lambda  = 200000;
lambda  = 1;
% S_train    = NormalizeFea(S_train);
W = SAE(X_tr', S_train', lambda)';

num_iter = 50;
for iter = 1:num_iter
    W = SAE(X_tr', (X_tr*W)', lambda)';
    disp(100 * iter / num_iter);
end
% test_x_weighted = X_tr * W * W';
% test_x = X_tr;
%% ^^^^^^ REPEAT THIS AND MEAN FOR ALL SIMILAR LABELS
S_train = X_tr * W;
S_te_est = X_te * W;

%%

dist     =  1 - (pdist2((S_te_est), (S_test_pos), 'cosine'));
dist     = zscore(dist);
HITK     = 1;
Y_hit5   = zeros(size(dist,1),HITK);
for i    = 1:size(dist,1)
    [sort_dist_i, I] = sort(dist(i,:),'descend');
    Y_hit5(i,:) = unseen_uniq_labels(I(1:HITK));
end

n=0;
for i  = 1:size(dist,1)
    if ismember(unseen_labels(i),Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy = n/size(dist,1);
fprintf('\n[1] bruh ZSL accuracy [V >>> S]: %.1f%%\n', zsl_accuracy*100);
%%
X_s = (S_te_est * W');
X_f = (W_old' \ X_s')';

%%
X_trf = seen_data_features;

%%
% test_idx = 40000; 
% img_test_feat = reshape(X_f(test_idx, :), 7, 7, 32, 1);

% test_idx_L = 40000; % L
% test_idx_Y = 75810; % Y
% test_idx_7 = 20000; % 7
% test_idx_W = 60010; % W
% img_feat_NE = X_tef(test_idx_L, :);
% img_feat_SE = X_tef(test_idx_Y, :);
% img_feat_NW = X_tef(test_idx_7, :);
% img_feat_SW = X_tef(test_idx_W, :);

test_idx_3 = 21000; % 3
test_idx_8 = 41075; % 8
test_idx_G = 101000; % G
test_idx_S = 281050; % S
% img_feat_NE = double(seen_data(test_idx_3, :));
% img_feat_SE = double(seen_data(test_idx_8, :));
% img_feat_NW = double(seen_data(test_idx_G, :));
% img_feat_SW = double(seen_data(test_idx_S, :));
img_feat_NE = X_seen(test_idx_3, :) * W_latent;
img_feat_SE = X_seen(test_idx_8, :) * W_latent;
img_feat_NW = X_seen(test_idx_G, :) * W_latent;
img_feat_SW = X_seen(test_idx_S, :) * W_latent;
% img_feat_NE = X_seen(test_idx_3, :);
% img_feat_SE = X_seen(test_idx_8, :);
% img_feat_NW = X_seen(test_idx_G, :);
% img_feat_SW = X_seen(test_idx_S, :);
% img_feat_NE = X_trf(test_idx_3, :);
% img_feat_SE = X_trf(test_idx_8, :);
% img_feat_NW = X_trf(test_idx_G, :);
% img_feat_SW = X_trf(test_idx_S, :);
% test = X_trf(test_idx_3, :);
% test0 = (W_feat' \ (X_trf(test_idx_3, :) * W_feat)')';
%%
img_feat_NE = (W_feat' \ (X_trf(test_idx_3, :) * W_feat * W_latent * W_latent')')';
img_feat_SE = (W_feat' \ (X_trf(test_idx_8, :) * W_feat * W_latent * W_latent')')';
img_feat_NW = (W_feat' \ (X_trf(test_idx_G, :) * W_feat * W_latent * W_latent')')';
img_feat_SW = (W_feat' \ (X_trf(test_idx_S, :) * W_feat * W_latent * W_latent')')';

% unseen_size = length(unseen_labels);
% img_feat_NE = X_f(randi(unseen_size), :);
% img_feat_SE = X_f(randi(unseen_size), :);
% img_feat_NW = X_f(randi(unseen_size), :);
% img_feat_SW = X_f(randi(unseen_size), :);
% img_feat_NN = X_f(randi(unseen_size), :);
% img_feat_SS = X_f(randi(unseen_size), :);
% img_feat_EE = X_f(randi(unseen_size), :);
% img_feat_WW = X_f(randi(unseen_size), :);
% img_feat_CC = X_f(randi(unseen_size), :);

% seen_size = length(seen_labels);
% img_feat_NE = X_trf(randi(seen_size), :);
% img_feat_SE = X_trf(randi(seen_size), :);
% img_feat_NW = X_trf(randi(seen_size), :);
% img_feat_SW = X_trf(randi(seen_size), :);
% img_feat_NN = X_trf(randi(seen_size), :);
% img_feat_SS = X_trf(randi(seen_size), :);
% img_feat_EE = X_trf(randi(seen_size), :);
% img_feat_WW = X_trf(randi(seen_size), :);
% img_feat_CC = X_trf(randi(seen_size), :);

% num_feat = 1568;
% img_feat_NE = randn(1, num_feat);
% img_feat_SE = randn(1, num_feat);
% img_feat_NW = randn(1, num_feat);
% img_feat_SW = randn(1, num_feat);
% img_feat_NN = randn(1, num_feat);
% img_feat_SS = randn(1, num_feat);
% img_feat_EE = randn(1, num_feat);
% img_feat_WW = randn(1, num_feat);
% img_feat_CC = randn(1, num_feat);

% img_feat_L = reshape(X_tef(test_idx_L, :), 7, 7, 32, 1);
% img_feat_Y = reshape(X_tef(test_idx_Y, :), 7, 7, 32, 1);
% img_feat_7 = reshape(X_tef(test_idx_7, :), 7, 7, 32, 1);
% img_feat_W = reshape(X_tef(test_idx_W, :), 7, 7, 32, 1);
%%
scale = 1;
left_feats  = scale * [img_feat_NE; img_feat_SE];
right_feats = scale * [img_feat_NW; img_feat_SW];
% left_feats  = scale * [img_feat_NE; img_feat_EE; img_feat_SE];
% cent_feats  = scale * [img_feat_NN; img_feat_CC; img_feat_SS];
% right_feats = scale * [img_feat_NW; img_feat_WW; img_feat_SW];
num_interp = 9;
npt_interp = linspace(0, 1, num_interp);

% left_feats(left_feats < 0) = 0;
% cent_feats(cent_feats < 0) = 0;
% right_feats(right_feats < 0) = 0;

left_feats = reshape(interp1(linspace(0, 1, size(left_feats, 1)), left_feats, npt_interp), 1, length(left_feats), num_interp);
% cent_feats = reshape(interp1(linspace(0, 1, size(cent_feats, 1)), cent_feats, npt_interp), 1, length(cent_feats), num_interp);
right_feats = reshape(interp1(linspace(0, 1, size(right_feats, 1)), right_feats, npt_interp), 1, length(left_feats), num_interp);

all_feats = [left_feats; right_feats];
% all_feats = [left_feats; cent_feats; right_feats];
all_feats = reshape(interp1(linspace(0, 1, size(all_feats, 1)), all_feats, npt_interp), num_interp*num_interp, length(left_feats));
%%
output_path = "data/data0/classify1/";
png_id = 420;
% pngs = zeros(dims, dims, num_interp*num_interp);
pngs = [];
png = [];
for fidx = 1:num_interp*num_interp
%     test = predict(decode_net, reshape(all_feats(fidx, :), 7, 7, 32, 1));
    test = predict(decode_net, reshape(all_feats(fidx, :) * W_latent', 1, 1, 26, 1));
%     test = predict(decode_net, reshape(all_feats(fidx, :), 1, 1, 26, 1));
%     test = reshape(all_feats(fidx, :), 28, 28);
    png = [png, imdisp(reshape(test, dims, dims), "surpress", 1)];
    if ~logical(mod(fidx, num_interp))
        pngs = [pngs; png];
        png = [];
    end
%     imwrite(png, sprintf("%s%d_%d.png", output_path, png_id, fidx - 1));
end
imwrite(pngs, sprintf("%s%d.png", output_path, png_id));

%%
test = predict(decode_net, img_test_feat);
imdisp(reshape(test, dims, dims));
% disp(unseen_labels(test_idx));

%%

% test = softmax(net.Layers(13, 1).Weights*X_trf' + net.Layers(13, 1).Bias);
% [~, testi] = max(test);

% test = softmax(net.Layers(13, 1).Weights*X_tef' + net.Layers(13, 1).Bias);
% [~, testi] = max(test);
% bruh = net.Layers(end, 1).Classes(testi);
% y_unseen = categorical([unseen_uniq_labels; ones(17, 1)]);
% bruh = y_unseen(testi);