function [ dati ] = loadDataFromDateOrIndex( list, date, window_size, prefixType )
%IMPORTFILE Summary of this function goes here
%   Detailed explanation goes here

%% load data starting from that file until window_size measurements are loaded or end is reached
dati.time = [];
dati.vasca = [];
dati.soffiatore = [];
index = 1;
if isdatetime(date)
    while list{index, 1} < date 
        index = index + 1;
    end
else
    assert(isnumeric(date));
    index = floor(date);
    assert(index > 0);
end

while (window_size > 0) && index <= size(list, 1)
    [ tmp, is_nan ] = importFile(list{index, 2}, prefixType);
    dati.time = cat(2, dati.time, tmp.time);
    dati.soffiatore = cat(2, dati.soffiatore, tmp.soffiatore);
    dati.vasca = cat(3, dati.vasca, tmp.vasca);
    window_size = window_size - size(tmp.time, 2);
    fprintf('parsed file %s : found %d nan values\n', list{index, 2}, sum(is_nan));
    index = index + 1;
end

end

