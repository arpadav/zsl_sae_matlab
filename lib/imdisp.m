function varargout = imdisp(X, varargin)
% imdisp -  ARPAD ATTILA VOROS
%           October 22, 2021
%           Displays a 2-dimensional matrix 0-255 (unless specified
%           otherwise), or 3-dimensional matrice
%               If 3-dimensional, only the first 3 channels are displayed
%               as RGB. If lacking last channel (B), replaced with 0's
%   INPUTS:     X - input matrix, 2 or 3-dimensional
%               varargin - 
%                   Suppress - supress display/imshow, only return image
%                   DispType - 
%                       What to display
%                       If none specified, display raw. However if complex,
%                       defaults to magnitude
%                       1. "abs", "mag", 1
%                           displays magnitude
%                       2. "angle", "phase", 2
%                           displays phase
%                   Transform -
%                   TRANSFORMS APPLIED IN ORDER OF INPUT!
%                       0. "log", 1
%                           transforms output with natural log:
%                           log(1 + abs(result))
%                       1. "log10", 2
%                           transforms output with log base 10:
%                           log10(1 + abs(result))
%                       2. "histeq", 3
%                           histogram equalization
%                       4. "histmatch", 4
%                           histogram matching with reference array
%                           - input any-dimensional array afterward
%                               e.g. (..."Transform", "histmatch", ref...)
%                               where ref is any-d reference array
%                       5. "median_offset", 5
%                           transforms output with median offset
%                           sqrt((result - median(result)).^2)
%                       6. "mean_offset", 6
%                           transforms output with mean offset
%                           sqrt((result - mean(result)).^2)
%                       7. "mode_offset", 7
%                           transforms output with mode offset
%                           sqrt((result - mode(result)).^2)
%               DispSetting -
%                       1. "inv", "invert", 1
%               Range - [min, max], where min and max are converted to 
%                       uint8. if none specified AND no histeq/histmatch,
%                       then defaults to [0, 255]
%               RGB -   if 2-dimensional, displays result on specified
%                       channel
%                       if 3-dimensional, displays ONLY selected channel
%                       1. "r", 1
%                           red channel
%                       2. "g", 2
%                           green channel
%                       3. "b", 3
%                           blue channel
%   OUTPUTS:    Y - output image, uint8
% 
%   An example function call:
%       output = imdisp(input, "DispType", "mag", "DispSetting", "inv", ...
%                       "Transform", "log", "histmatch", histmatch_ref, ...
%                       "median_offset");
%   An EQUIVALENT call to the one above
%       output = imdisp(input, "DispSetting", 1, "disptype", "abs", ...
%                       "tRANSFORM", 1, "histmatch", histmatch_ref, ...
%                       5);

% check dimensions
if ndims(X) < 2 || ndims(X) > 3
    error("Input must be a 2 or 3-dimensional matrice.");
end

% varargin categories
var_cat = [ "Surpress", ...
            "DispType", ...
            "Transform", ...
            "DispSetting", ...
            "Range", ...
            "ScaleChannels", ...
            "RGB" ...
            ];
% varargin varargin subcategories
var_subcat = {  {{}, {"none"}}, ...
                {{"mag", "abs", 1}, {"angle", "phase", 2}}, ...
                {{"log", 1}, {"log10", 2}, ...
                    {"histeq", 3}, {"histmatch", 4} ...
                    {"median_offset", 5}, {"mean_offset", 6}, {"mode_offset", 7}}, ...
                {{"inv", "invert", 1}}, ...
                {{}, {"none"}}, ...
                {{}, {"none"}}, ...
                {{"r", 1}, {"g", 2}, {"b", 3}} ...
                };
% default values if none specified
var_defaults = {{"none"}, ... 
                {"none"}, ...
                {"none"}, ...
                {"none"}, ... 
                {[0 255]}, ... % {[0 255]} {"none"}
                {"none"}, ... 
                {"none"} ...
                }; %#ok<*STRSCALR> 
% number of inputs (including subcategory) after category
% 1 unless specified otherwise
var_num_additional_input = {{[3, 4], 1}};
% mutually-exclusive display setting
var_uniq = logical([1, 1, 0, 0, 1, 1, 1]);

% parse varargin, get struct
[varg, vargord] = parsevarargin(varargin, var_cat, var_subcat, var_defaults, var_num_additional_input, var_uniq);

% clip Range 0 -> 255 if OOB
specified_range = false(1);
if numel(varg.Range) == 2
    varg.Range = double(uint8(varg.Range));
    specified_range = true(1);
end

% size of input
[xh, xw, xc] = size(X);
if ~isreal(X)
    % complex values, display magnitude if no DispType specified
    if ~varg.DispType
        varg.DispType = true(1);
        varg.mag = true(1);
    end
end

% turn input into double for future operations
X = double(X);
% initialize output
Y = uint8(zeros(xh, xw, xc));
% loop through each channel
for c = 1:xc
    if varg.DispType
        if varg.mag
            x = abs(X(:, :, c));
        elseif varg.angle
            x = angle(X(:, :, c));
        else
            x = X(:, :, c);
        end
    else
        x = X(:, :, c);
    end
    % preprocessing display options
    if varg.Transform
        for transform = vargord.Transform
            switch transform
                case "log"
                    x = log(1 + abs(x));
                case "log10"
                    x = log10(1 + abs(x));
                case "histeq"
                    x = histeq(x ./ max(x, [], 'all'));
                case "histmatch"
                    if size(varg.histmatch, 3) == 1
                        x = imhistmatch(x ./ max(x, [], 'all'), varg.histmatch ./ max(varg.histmatch, [], 'all'));
                    elseif size(varg.histmatch, 3) == 3
                        x = imhistmatch(x ./ max(x, [], 'all'), varg.histmatch(:, :, c) ./ max(varg.histmatch(:, :, c), [], 'all'));
                    end
                case "median_offset"
                    x = sqrt((x - median(x, 'all')).^2);
                case "mean_offset"
                    x = sqrt((x - mean(x, 'all')).^2);
                case "mode_offset"
                    x = sqrt((x - mode(x, 'all')).^2);
                otherwise
            end
        end
    end
    % if range is specified
    if specified_range
        if min(x, [], 'all') ~= max(x, [], 'all')
            x = x - min(x, [], 'all');
            Y(:, :, c) = uint8(((varg.Range(2) - varg.Range(1)) * x / max(x, [], 'all')) + varg.Range(1));
        else
            % all the same number. if non-zero, scale to max
            if min(x, [], 'all')
                x = varg.Range(2) * ones(size(x));
                Y(:, :, c) = uint8(x);
            else
                x = varg.Range(1) * ones(size(x));
                Y(:, :, c) = uint8(x);
            end
        end
    % otherwise, no specified range
    else
        Y(:, :, c) = uint8(x);
    end
end
% display the image
if varg.inv
    Y = 255 - Y;
end

% scale all channels via mag
if varg.ScaleChannels && specified_range
    Y = double(Y);
    Y_mag = zeros(xh, xw);
    for c = 1:xc
        Y_mag = Y_mag + Y(:, :, c).^2;
    end
    Y_mag = sqrt(Y_mag);

    % do scale
    Y_mag_scaled = Y_mag;
    if min(Y_mag_scaled, [], 'all') ~= max(Y_mag_scaled, [], 'all')
        Y_mag_scaled = Y_mag_scaled - min(Y_mag_scaled, [], 'all');
        Y_mag_scaled = ((varg.Range(2) - varg.Range(1)) * Y_mag_scaled / max(Y_mag_scaled, [], 'all')) + varg.Range(1);
    else
        % all the same number. if non-zero, scale to max
        if min(Y_mag_scaled, [], 'all')
            Y_mag_scaled = varg.Range(2) * ones(size(x));
        else
            Y_mag_scaled = varg.Range(1) * ones(size(x));
        end
    end
    
    % do da scale again!
    for c = 1:xc
        Y(:, :, c) = Y(:, :, c) .* sqrt(Y_mag_scaled) ./ sqrt(Y_mag);
    end
    Y = uint8(Y);
end

if varg.RGB
    % 3-dimensional
    if xc > 1
        switch vargord.RGB
            case "r"
                Y(:, :, 2:3) = 0;
            case "g"
                Y(:, :, 1:2:3) = 0;
            case "b"
                Y(:, :, 1:2) = 0;
            otherwise
        end
    % 2-dimensional
    else
        switch vargord.RGB
            case "r"
                Y = cat(3, Y, zeros(xh, xw, 2));
            case "g"
                Y = cat(3, zeros(xh, xw, 1), Y, zeros(xh, xw, 1));
            case "b"
                Y = cat(3, zeros(xh, xw, 2), Y);
            otherwise
        end
    end
end
if ~varg.Surpress
    imshow(Y);
end
varargout{1} = Y;
end