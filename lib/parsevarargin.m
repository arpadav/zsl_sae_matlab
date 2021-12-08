function [val_struct, varargout] = parsevarargin(varargin_input, categories, subcategories, defaults, num_additional_input_arr, uniq)
% parsevarargin -   ARPAD ATTILA VOROS
%                   October 19, 2021
%   INPUTS:     categories - string array of main categories
%               subcategories - 2-3x nested cell array of potential
%                               subcategories for each category
%                               1st dim - cells for each category
%                               2nd dim - within each subcat, cells for 
%                                         each equivalent option
%                               3rd dim - optional, ordered input
%                   defaults - default values for varargin. cell array size
%                              categories. each category has default value.
%                              "none" string indicates falsehood
%                   num_additional_input_arr - number of additional inputs 
%                                              for a specified subcategory
%                   uniq - categories which have mutually exclusive
%                          subcategories
%   OUTPUTS:    parsed - parsed input into struct format
% 
% EXAMPLE OF INPUT PARAEMTER FORMAT
% % varargin categories
% categories = ["DispType", ...
%               "Transform", ...
%               "DispSetting", ...
%               "Range", ...
%               "RGB" ...
%               ];
% % varargin subcategories
% subcategories = { {{"mag", "abs", 1}, {"angle", "phase", 2}}, ...
%                   {{"log", 1}, {"log10", 2}, ...
%                    {"histeq", 3}, {"histmatch", 4} ...
%                    {"median_offset", 5}, {"mean_offset", 6}, {"mode_offset", 7}}, ...
%                   {{"inv", "invert", 1}}, ...
%                   {{}}, ...
%                   {{"r", 1}, {"g", 2}, {"b", 3}} ...
%                   };
% % default values if none specified
% defaults = {  {"none"}, ...
%               {"none"}, ...
%               {"none"}, ...
%               {[0 255]}, ...
%               {"none"} ...
%               }; %#ok<*STRSCALR> 
% % number of inputs (including subcategory) after category
% % 1 unless specified otherwise
% num_additional_input_arr = {{[2, 4], 1}};
% % mutually-exclusive display setting
% uniq = logical([1, 0, 0, 1, 1]);

% parsing varargin for display parameters
% number of categories
num_categories = length(categories);
% number irregular input sizes
idx_numinput_len = length(num_additional_input_arr);
% number of total possible fields (include main categories incase
% non qualitative response)
field_count = num_categories;
% qualitative input, i.e. not numeric
qualidx = 1:num_categories;
for cat_idx = 1:num_categories
    % categories without subcategories, i.e. quantitative
    if isempty(subcategories{cat_idx}{1}) % numel(subcategories{cat_idx}) == 1 && 
        % if quantitative, remove from the list
        qualidx(qualidx == cat_idx) = [];
    else
        % add all subcategories to field count
        field_count = field_count + numel(subcategories{cat_idx});
    end
end

% START FSM
state = 0;
prev_state = 0;
vidx = 0;
arg = 0;
subcat_idx = 0;
num_additional_inputs = 0;

% varargin length
varargin_length = length(varargin_input);
% for creating varargin struct
varargin_field_index = 1;
varargin_field_id = string(cellfun(@(x) "", cell(1, field_count), 'UniformOutput', false));
% populate field ids
for cat_idx = 1:num_categories
    varargin_field_id(varargin_field_index) = categories(cat_idx);
    varargin_field_index = varargin_field_index + 1;
    for subcat_idx = 1:numel(subcategories{cat_idx})
        % if there are subcategories, append to field id
        if ~isempty(subcategories{cat_idx}{subcat_idx})
            % field id is first option
            varargin_field_id(varargin_field_index) = lower(string(subcategories{cat_idx}{subcat_idx}{1}));
            varargin_field_index = varargin_field_index + 1;
        elseif subcat_idx == 1
            % first index is empty, meaning QUANTITATIVE or NONE so leave
            break;
        end
    end
end
% initialize field values
varargin_field_value = cell(1, field_count);
% initialize subcat field order
varargin_subcat_field_order = zeros(1, field_count);
% FSM, parsing varargin like regex
while vidx < varargin_length + 1
    switch state
        % checking for category
        case 0
            if prev_state == 0
                vidx = vidx + 1;
                if vidx > varargin_length
                    break;
                end
                % get new input
                try
                    arg = lower(varargin_input{vidx});
                catch
                    arg = varargin_input{vidx};
                end
                if ismember(arg, lower(categories))
                    % update category
                    cat_idx = arg == lower(categories);
                    % move states
                    prev_state = state;
                    state = 1;
                else
                    error("Error parsing display settings: check spelling.");
                end
            elseif ismember(arg, lower(categories))
                % update category
                cat_idx = arg == lower(categories);
                % move states
                prev_state = state;
                state = 1;
            else
                error("Error parsing display settings: check spelling.");
            end
        % checking new instance of subcategory
        case 1
            vidx = vidx + 1;
            if vidx > varargin_length
                break;
            end
            % get new input
            try
                arg = lower(varargin_input{vidx});
            catch
                arg = varargin_input{vidx};
            end
            % accumulate all subcategory possibilities
            subcat_possibilities = cellfun(@(x) x(:), subcategories{cat_idx}(:), 'UniformOutput', false);
            % find if empty subcategories allowed - i.e. ANY input allowed,
            % so quantitative possibility
            subcat_empties = false(1);
            for subcat_poss_idx = 1:numel(subcat_possibilities)
                if isempty(subcat_possibilities{subcat_poss_idx})
                    subcat_empties = true(1);
                    break;
                end
            end
            subcat_possibilities = vertcat(subcat_possibilities{:});
            % input argument is category
            if isstring(arg) && ismember(arg, lower(categories))
                prev_state = state;
                state = 0;
            % input argument is a potential subcategory
            elseif isstring(arg) && ismember(lower(arg), string(subcat_possibilities)) % any(cellfun(@(x) isequal(lower(x), arg), subcat_possibilities))
                % quantitative possibility, but equals other subcategory
                % possibilities. if none, set false, otherwise add the
                % subcategory to the lists (previously ignored)
                if subcat_empties && isequal(lower(arg), "none")
                    varargin_field_value{varargin_field_id == categories(cat_idx)} = false(1);
                    prev_state = state;
                    state = 1;
                else
                    if subcat_empties
                        % quantitative OR some other thing which was
                        % previously ignored, now appended
                        field_count = field_count + 1;
                        varargin_field_id(field_count) = lower(arg);
                        varargin_field_value{field_count} = [];
                    end
                    % if category is empty, set it to true
                    if isempty(varargin_field_value{varargin_field_id == categories(cat_idx)})
                        varargin_field_value{varargin_field_id == categories(cat_idx)} = true(1);
                    end
                    % find subcategory
                    subcat_idx = 1;
                    for subcat_parse_idx = 1:numel(subcategories{cat_idx})
                        if max(cellfun(@(x) isequal(lower(x), arg), subcategories{cat_idx}{subcat_parse_idx}))
                            subcat_idx = subcat_parse_idx;
                        end
                    end
                    % update subcat field order
                    varargin_subcat_field_order(varargin_field_id == subcategories{cat_idx}{subcat_idx}{1}) = vidx;
                    % find the number of additional inputs given the subcat
                    num_additional_inputs = 0;
                    for numinput_idx = 1:idx_numinput_len
                        if find(cat_idx) == num_additional_input_arr{numinput_idx}{1}(1) && subcat_idx == num_additional_input_arr{numinput_idx}{1}(2)
                            num_additional_inputs = num_additional_input_arr{numinput_idx}{2};
                        end
                    end
                    if num_additional_inputs > 0
                        % qualitative found, next is quantitative
                        vidx = vidx + 1;
                        if vidx > varargin_length
                            break;
                        end
                        % get new input
                        try
                            arg = lower(varargin_input{vidx});
                        catch
                            arg = varargin_input{vidx};
                        end
                        prev_state = state;
                        state = 2;
                    else
                        % qualitative
                        varargin_field_value{varargin_field_id == subcategories{cat_idx}{subcat_idx}{1}} = true(1);
                        if uniq(cat_idx)
                            prev_state = state;
                            state = 3;
                        else
                            prev_state = state;
                            state = 1;
                        end
                    end
                end
            % no subcategory + quantitative due to empty possibilities
            elseif subcat_empties
                % quantitative, therefore no qualitative subcategory
                subcat_idx = 1;
                % find the number of additional inputs
                num_additional_inputs = 1;
                for numinput_idx = 1:idx_numinput_len
                    if find(cat_idx) == num_additional_input_arr{numinput_idx}{1}(1) && subcat_idx == num_additional_input_arr{numinput_idx}{1}(2)
                        num_additional_inputs = num_additional_input_arr{numinput_idx}{2};
                    end
                end
                % move states
                prev_state = state;
                state = 2;
            else
                error("Error parsing display settings: check formatting.");
            end
        % looping through fixed number of additional inputs
        case 2
            % first entered looping of additional inputs
            if prev_state == 1
                if numel(subcategories{cat_idx}{subcat_idx})
                    varargin_field_value{varargin_field_id == subcategories{cat_idx}{subcat_idx}{1}} = ...
                        [varargin_field_value{varargin_field_id == subcategories{cat_idx}{subcat_idx}{1}}, {arg}];
                else
                    varargin_field_value{varargin_field_id == categories(cat_idx)} = ... 
                        [varargin_field_value{varargin_field_id == categories(cat_idx)}, {arg}];
                end
                
                num_additional_inputs = num_additional_inputs - 1;
                % change state
                if num_additional_inputs == 0
                    prev_state = state;
                    state = 1;
                else
                    prev_state = state;
                    state = 2;
                end
            % has been looping of additional inputs
            elseif prev_state == 2
                vidx = vidx + 1;
                if vidx > varargin_length
                    break;
                end
                % get new input
                try
                    arg = lower(varargin_input{vidx});
                catch
                    arg = varargin_input{vidx};
                end
                % append argument to output
                varargin_field_value{varargin_field_id == subcategories{cat_idx}{subcat_idx}{1}} = ...
                    [varargin_field_value{varargin_field_id == subcategories{cat_idx}{subcat_idx}{1}}, {arg}];
                % update subcategory field order, if not already
                if varargin_subcat_field_order(varargin_field_id == subcategories{cat_idx}{subcat_idx}{1}) == 0
                    varargin_subcat_field_order(varargin_field_id == subcategories{cat_idx}{subcat_idx}{1}) = vidx;
                end
                % update number of additional inputs
                num_additional_inputs = num_additional_inputs - 1;
                % change state
                if num_additional_inputs == 0
                    prev_state = state;
                    state = 1;
                else
                    prev_state = state;
                    state = 2;
                end
            else
                error("Illegal FSM configuration.");
            end
        % if unique subcategory, set all other subcategories of same
        % category to false
        case 3
            for subcat_parse_idx = 1:numel(subcategories{cat_idx})
                if subcat_parse_idx ~= subcat_idx
                    if ~isempty(subcategories{cat_idx}{subcat_parse_idx}) && any(varargin_field_id == subcategories{cat_idx}{subcat_parse_idx}{1})
                        varargin_field_value{varargin_field_id == subcategories{cat_idx}{subcat_parse_idx}{1}} = [];
                        varargin_subcat_field_order(varargin_field_id == subcategories{cat_idx}{subcat_parse_idx}{1}) = 0;
                    end
                end
            end
            prev_state = state;
            state = 1;            
        otherwise
            state = 0;
    end
end
% transform subcat field order properly
ord_cell = cell(num_categories, 1);
vidx = 1;
for cat_idx = 1:num_categories
    num_subcat = numel(subcategories{cat_idx});
    if isempty(subcategories{cat_idx}{1}) % num_subcat == 1 && 
        num_subcat = 0;
    end
    idx_range = vidx + (0:num_subcat);
    vec = varargin_subcat_field_order(idx_range);
    if length(idx_range) > 1
        ord_mask = [false(1), vec(2:end) > 0];
        [~, ord_idx] = sort(vec(ord_mask));
        if ~isempty(ord_idx)
            ord_cell{cat_idx} = varargin_field_id(idx_range(ord_mask));
            ord_cell{cat_idx} = ord_cell{cat_idx}(ord_idx);
        end
    else
        ord_cell{cat_idx} = [];
    end
    vidx = idx_range(end) + 1; 
end
% if empty, use default
cat_idx = 0;
for varargin_field_index = 1:length(varargin_field_id)
    % find the indicies
    if ismember(varargin_field_id(varargin_field_index), categories)
        cat_idx = cat_idx + 1;
        subcat_idx = 0;
        def_vals = defaults{cat_idx};
    else
        subcat_idx = subcat_idx + 1;
    end
    % if no value used, append defaults to field values
    if isempty(varargin_field_value{varargin_field_index})
        if ~subcat_idx
            % a category that is empty
            if length(def_vals) == 1 && isequal(def_vals{1}, "none")
                % whole category is FALSE if default is 1x1 "none"
                varargin_field_value{varargin_field_index} = false(1);
            end
            % if STILL EMPTY, then there is a value to append
            if isempty(varargin_field_value{varargin_field_index})
                % difference between quantitative and qualitative
                if isempty(subcategories{cat_idx}{1}) % numel(subcategories{cat_idx}) == 1 && 
                    % if qualitative & NO subcategory values, set category
                    % field value to false. otherwise, quantitative, so
                    % must have DEFAULT DATA
                    if ismember(cat_idx, qualidx)
                        % set value to false since no subcategories
                        varargin_field_value{varargin_field_index} = false(1);
                    else
                        % set value to default value since no subcategories
                        varargin_field_value{varargin_field_index} = defaults{cat_idx}{1};
                    end
                else
                    % set value to true if there are default subcategories
                    varargin_field_value{varargin_field_index} = true(1);
                end
            end
        else
            % set field values accordingly
            % overwrite default value array, depending on whether unique
            % or not. use first index if unique
            if uniq(cat_idx)
                def_vals_overwrite = def_vals(1);
            else
                def_vals_overwrite = def_vals;
            end
            % loop through default values
            for def_vals_idx = 1:length(def_vals_overwrite)
                % if qualitative, ALL struct values must be boolean
                if ismember(cat_idx, qualidx)
                    % if equal to the index of the subcategory, then set
                    % field value to true and ALL OTHERS to false
                    if isequal(def_vals{def_vals_idx}, varargin_field_id(varargin_field_index))
                        varargin_field_value{varargin_field_index} = true(1);
                    else
                        varargin_field_value{varargin_field_index} = false(1);
                    end
                % if quantitative, chosen struct value must be DATA, others
                % default to false
                else
                    % if equal to the index of the subcategory, then set
                    % field value to default and ALL OTHERS to false
                    if isequal(def_vals{def_vals_idx}, varargin_field_id(varargin_field_index))
                        varargin_field_value{varargin_field_index} = defaults{cat_idx}{subcat_idx};
                    else
                        varargin_field_value{varargin_field_index} = false(1);
                    end
                end
            end
        end
    elseif iscell(varargin_field_value{varargin_field_index}) && numel(varargin_field_value{varargin_field_index}) == 1
        varargin_field_value{varargin_field_index} = varargin_field_value{varargin_field_index}{1};
    end
end
% create struct for parsing
struct_labels = cellstr(varargin_field_id);
varargin_field_value = varargin_field_value(:);
val_struct = cell2struct(varargin_field_value, struct_labels, 1);
ord_struct = cell2struct(ord_cell, cellstr(categories), 1);
varargout{1} = ord_struct;
end