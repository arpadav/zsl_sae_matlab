%% seperate data and labels bc logical for less storage
% load('data/alphabet_uppercase.mat');
% chars_d = logical(chars(:, 2:end));
% chars_l = chars(:, 1);

load('data/digits.mat');
digits_d = logical(digits(:, 2:end));
digits_l = digits(:, 1);

%% A-Z mnist to .mat logical (offset 127 (<= is 0, > is 1))
load('data/uint8/alphabet_uppercase.mat');
old_chars = chars;
chars = zeros(size(old_chars));

chars(:, 1) = old_chars(:, 1);
chars(:, 2:end) = double(old_chars(:, 2:end) > 127);


%% 0-9 mnist to .mat (offset 127 (<= is 0, > is 1))
load('data/uint8/digits.mat');
% digits(:, 2:end) = double(uint8(digits(:, 2:end) * 255));
old_digits = digits;
digits = zeros(size(old_digits));

digits(:, 1) = old_digits(:, 1);
digits(:, 2:end) = double(old_digits(:, 2:end) > 127);

%% 0-9 A-Z label to char encoding
data_labels = 0:(10 + 25);
data_unicode = [48:(48 + 9), 65:(65 + 25)];

labels_ref_table = [data_labels' data_unicode'];

%% resize saved figures
dp = "data/data1/classify1/538 preds2/";
num_iter = 25;
% numpixcut_lr = 90;
% numpixcut_top = 1;
% numpixcut_bot = 92;
% for iter = 1:num_iter
%     png = imread(strcat(dp, "p", string(iter), ".png"));
% %     imshow(png(numpixcut_top:end-numpixcut_bot, numpixcut_lr:end-numpixcut_lr, :));
% %     disp("bruh");
% %     imwrite(png(numpixcut_top:end-numpixcut_bot, numpixcut_lr:end-numpixcut_lr, :), strcat(dp, "p", string(iter), ".png"));
% end

%% combine saved figures
dp = "data/data0/classify1/838 preds2/";
num_iter = 25;
png = [];
pngs = [];
for iter = 1:num_iter
%     png = [png, imread(strcat(dp, "p", string(iter), ".png"))];
    if ~logical(mod(iter, 5))
        pngs = [pngs; png];
        png = [];
    end
%     imshow(png(numpixcut_top:end-numpixcut_bot, numpixcut_lr:end-numpixcut_lr, :));
%     disp("bruh");
%     imwrite(png(numpixcut_top:end-numpixcut_bot, numpixcut_lr:end-numpixcut_lr, :), strcat(dp, "p", string(iter), ".png"));
end
% imwrite(pngs, strcat(dp, "838comb2.png"));