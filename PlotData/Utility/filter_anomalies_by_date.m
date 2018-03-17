function [ filtered ] = filter_anomalies_by_date(anomalie, begin_date, end_date)
filtered = [];
for i=1:size(anomalie, 1)
    % not ( window end before begin of event, or window begin after event end )
    if ~ (end_date < anomalie{i, 3}{1} || begin_date > anomalie{i, 3}{2})
        tmp_begin = max(begin_date, anomalie{i, 3}{1});
        tmp_end = min(end_date, anomalie{i,3}{2});
        filtered = [filtered; {anomalie{i, 1}, anomalie{i, 2}, {tmp_begin, tmp_end}}];
    end
end

end

