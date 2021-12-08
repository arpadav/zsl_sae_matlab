function refs = get_refs(lref, lbl, varargin)
% 
lbls = cellfun(@(l) l(1), lref);
refs = lref{lbls == lbl}(2:end);
if ~isempty(varargin)
    refs = refs(1);
end
end