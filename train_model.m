%%% arpad attila voros
%% pull data
clc; clear;
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

% redo labels, get reference arrays
[labels, lref] = account_similar_chars(labels, sim_data, ref_idx, lrefmat);

%% rand sample test labels + data for training
% get unique labels
uniq_labels = unique(labels);
num_uniq_labels = length(uniq_labels);

% ratio of seen and unseen labels for ZSL
seen_ratio = 0.5;
num_unseen = round(num_uniq_labels / (seen_ratio + 1));
unseen_uniq_labels = sort(datasample(uniq_labels, num_unseen, 'Replace', false));
unseen_labels_hotidx = sum(labels' == unseen_uniq_labels)';
seen_data =     data(find(~unseen_labels_hotidx), :);   %#ok<FNDSB> 
unseen_data =   data(find(unseen_labels_hotidx), :);    %#ok<FNDSB> 
seen_labels =   labels(find(~unseen_labels_hotidx), :); %#ok<FNDSB> 
unseen_labels = labels(find(unseen_labels_hotidx), :);  %#ok<FNDSB> 

%% datapath for processed data
datapath = "data/";
datafolder = "data0";
% save to datapath for training
save(strcat(datapath, datafolder, "/", datafolder, ".mat"), "dims", "lref", ...
    "unseen_uniq_labels", "seen_data", "unseen_data", "seen_labels", "unseen_labels", ...
    "datapath", "datafolder");

%% actually train network to get features
clear; clc;
load("data/data1/data1.mat");

xtrain = reshape(seen_data', dims, dims, 1, size(seen_data, 1));
ytrain = categorical(seen_labels);
seen_uniq_labels = unique(seen_labels);
num_final_layer = length(seen_uniq_labels);

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

net = trainNetwork(xtrain, ytrain, layers, options);
save(strcat(datapath, datafolder, "/classify1/net_classify1.mat"), "net");


%% get features
feature_layer = 'relu_3';
seen_data_features = activations(net, reshape(seen_data', dims, dims, 1, size(seen_data, 1)), feature_layer, 'OutputAs', 'rows');
unseen_data_features = activations(net, reshape(unseen_data', dims, dims, 1, size(unseen_data, 1)), feature_layer, 'OutputAs', 'rows');
feature_size = [7, 7, 32];

% save to datapath for decoder training
save(strcat(datapath, datafolder, "/classify1/seendata_f_classify1.mat"), "seen_data_features");
save(strcat(datapath, datafolder, "/classify1/unseendata_f_classify1.mat"), "unseen_data_features");

%% train decoder on FEATURES ONLY
clearvars -except datafolder datapath dims seen_data seen_data_features feature_size

layers_decoder = [
    imageInputLayer(feature_size,"Name","imageinput","Normalization","none")

    transposedConv2dLayer([3 3], 32,"Name","transposed-conv_1","Cropping","same")
    batchNormalizationLayer

    transposedConv2dLayer([3 3], 16,"Name","transposed-conv_2","Cropping","same")
    batchNormalizationLayer

    transposedConv2dLayer([3 3], 8,"Name","transposed-conv_3","Cropping","same")
    fullyConnectedLayer(dims*dims,"Name","fc")
    sigmoidLayer
    regressionLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

decode_net = trainNetwork(reshape(seen_data_features', feature_size(1), feature_size(2), feature_size(3), size(seen_data_features, 1)), ...
    double(seen_data), layers_decoder, options);
save(strcat(datapath, datafolder, "/classify1/decode_net_classify1.mat"), "decode_net");

%% train features AFTER weights are determined (see eval_model.m. load in W_feat) for semantic --> feature
% then use other network feature --> visual
feature_size = [1, 1, 26];
% reps = 2;
% latent_feature_size = [reps, reps, 26];

X_seen = X_seen_feat * W_feat;
% X_seen_star = X_seen * W_feat';

% X_seen_rep = repmat(X_seen, 1, reps*reps);
% X_seen_rep = seen_data_features * W_feat;
dims = 28;
% clearvars -except datafolder datapath dims seen_data seen_data_features feature_size
%%
layers_decoder = [
    imageInputLayer(feature_size, "Name","imageinput","Normalization","none")
    fullyConnectedLayer(dims*4,"Name","fc1")

    transposedConv2dLayer([3 3], 16, "Name","transposed-conv_1","Cropping","same")
    batchNormalizationLayer

    transposedConv2dLayer([3 3], 32, "Name","transposed-conv_2","Cropping","same")
    batchNormalizationLayer

    transposedConv2dLayer([3 3], 64, "Name","transposed-conv_3","Cropping","same")
    fullyConnectedLayer(dims*dims*4,"Name","fc2")
    fullyConnectedLayer(dims*dims,"Name","fc3")
    sigmoidLayer
    regressionLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

decode_net = trainNetwork(reshape(X_seen, feature_size(1), feature_size(2), feature_size(3), size(X_seen, 1)), ...
    double(seen_data), layers_decoder, options);

save(strcat(datapath, datafolder, "/classify1/decode_net_classify1_NEW_SB.mat"), "decode_net");

%% test on mult net
feature_size = [1, 1, 26];
X_in = reshape(X_seen, feature_size(1), feature_size(2), feature_size(3), size(X_seen, 1));

ds_X_in = arrayDatastore(X_in);
ds_W_feat = arrayDatastore(W_feat);
ds_input = combine(ds_X_in, ds_W_feat);
%%
options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');
decodem_net_sb = trainNetwork(ds_input, double(seen_data), decodem_lg_sb, options);

%% test combine ds
ds_X_in_trunc = arrayDatastore(X_in(:, :, :, 1:100), 'IterationDimension', 4);
% ds_W_feat_exp = arrayDatastore(reshape(repmat(W_feat, 1, 1, 100), 1, size(W_feat, 1), size(W_feat, 2), 100), 'IterationDimension', 4);
ds_W_feat_exp = arrayDatastore(reshape(W_feat, 1, size(W_feat, 1), size(W_feat, 2)));
ds_output_test = arrayDatastore(double(seen_data(1:100, :)), 'IterationDimension', 1);
ds_input_test = combine(ds_X_in_trunc, ds_W_feat_exp, ds_output_test);
%%
ds_X_in_trunc = arrayDatastore(X_in, 'IterationDimension', 4);
% ds_W_feat_exp = arrayDatastore(reshape(repmat(W_feat, 1, 1, 100), 1, size(W_feat, 1), size(W_feat, 2), 100), 'IterationDimension', 4);
ds_W_feat_exp = arrayDatastore(reshape(W_feat, 1, size(W_feat, 1), size(W_feat, 2)));
ds_output_test = arrayDatastore(double(seen_data), 'IterationDimension', 1);
ds_input_test = combine(ds_X_in_trunc, ds_W_feat_exp, ds_output_test);
%%
options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

decodem_net_sb_test = trainNetwork(ds_input_test, decodem_lg_sb, options);