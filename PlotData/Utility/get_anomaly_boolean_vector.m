function [ bool_vector ] = get_anomaly_boolean_vector(items, anomalies)
    bool_vector = logical(zeros(size(items, 2), 1));
    if isempty(anomalies) || isempty(items)
        return;
    end
    
    for item_index=1:size(items, 2)
        for anomaly_index=1:size(anomalies, 1)
            begin_date = anomalies{anomaly_index, 3}{1};
            end_date = anomalies{anomaly_index, 3}{2};
            bool_vector(item_index) = bool_vector(item_index) || ((items(item_index) >= begin_date) && (items(item_index) <= end_date));
        end
    end
end

