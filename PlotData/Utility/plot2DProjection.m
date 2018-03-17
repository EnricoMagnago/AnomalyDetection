function plot2DProjection(f, files_list, prefixType, dataType, vascaType, anomalie, date)
	figures_number = numel(f);
    window_size = -1;
    for i=1:figures_number
        if ishandle(f(i))
            window_size = get(f(i), 'UserData');
            clf(f(i));
        end
    end
    if window_size == -1
        exit(1);
    end

    dati = loadDataFromDateOrIndex(files_list, date, window_size, prefixType);
    window_size = size(dati.time, 2);

    current_anomalies = filter_anomalies_by_date(anomalie, dati.time(1), dati.time(size(dati.time, 2)));
    is_anomaly_vector = get_anomaly_boolean_vector(dati.time, current_anomalies);

    for i=1:figures_number
        if ishandle(f(i))
            set(f(i), 'UserData', window_size);
            set(f(i), 'NumberTitle', 'off', 'Name', sprintf('%d: from %s to %s',...
                size(dati.time, 2), datestr(dati.time(1)), datestr(dati.time(size(dati.time, 2)))));
        end
    end

    colors = ['r', 'g', 'b'];
    %portata,sst,ossigeno,ammoniaca,nitrati,valvola
    for i=1:6
        if ~ishandle(f(i))
            continue;
        end
        uicontrol('Parent',f(i),'Style','pushbutton','Units','normalized','Position', [0.01 0.85 0.05 0.1],...
            'String', 'Date',...
            'Callback', {@startDateCallBack, f, files_list, prefixType, dataType, vascaType, anomalie});

        uicontrol('Parent',f(i),'Style','pushbutton','Units','normalized','Position', [0.01 0.70 0.05 0.10],...
            'String', strcat('ws ', num2str(window_size)),...
            'Callback', {@windowSizeCallBack, f, files_list, prefixType, dataType, vascaType, anomalie, date, window_size});

        %position in the subplot grid
        z = 1;
        % set current figure
        figure(f(i));
        % project on 2 sensors: i-th, j-th
        for j=1:6
            if j ~= i
                subplot(2,3,z);
                hold on;
                grid on;
                grid minor;
                %title([vascaType{i,1} ' ' vascaType{j,1}]);
                for v = 1:3
                    correct_j = dati.vasca(v,j, ~is_anomaly_vector); % dati.vasca(v,j,:)
                    correct_i = dati.vasca(v,i, ~is_anomaly_vector); % dati.vasca(v,i,:)
                    correct_time = dati.time(1, ~is_anomaly_vector);
                    anomalies_j = dati.vasca(v,j, is_anomaly_vector);
                    anomalies_i = dati.vasca(v,i, is_anomaly_vector);
                    anomalies_time = dati.time(1, is_anomaly_vector);

                    plot(reshape(correct_j, 1, numel(correct_time)), reshape(correct_i, 1, numel(correct_time)), 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', colors(v));
                    plot(reshape(anomalies_j, 1, numel(anomalies_time)), reshape(anomalies_i, 1, numel(anomalies_time)), 'x', 'MarkerSize', 4, 'MarkerFaceColor', colors(v), 'MarkerEdgeColor', colors(v));
                end
                xlabel(vascaType{j,1});
                ylabel(vascaType{i,1});
                z = z + 1;
            end
        end
        % plot i-th sensors over time
        subplot(2,3,6);
        hold on;
        %grid on;
        %grid minor;
        for v = 1:3
            plot(dati.time,reshape(dati.vasca(v,i,:), 1, numel(dati.time)), colors(v));
        end
        % xlabel('time');
        ylabel(vascaType{i,1});
        yl = ylim;
        % add lines to highlight events
        y_offset = -yl(2)/20;
        for j=1:size(current_anomalies, 1)
            begin_time = current_anomalies{j, 3}{1};
            end_time = current_anomalies{j, 3}{2};
            x = [begin_time begin_time end_time end_time begin_time];
            for k=1:size(current_anomalies{j, 2}, 2)
                plot(x, [y_offset y_offset y_offset y_offset y_offset], colors(current_anomalies{j, 2}(k)), 'LineWidth', 1);
                y_offset = y_offset - yl(2)/40;
            end
        end
        ylim([y_offset, inf]);

        h = legend('vasca 1','vasca 2','vasca 3');
        rect = [0.0, 0.5, 0.1, 0.1];
        set(h, 'Position', rect);
        legend('boxoff');
    end
    
    % plot frequencies.
    if ishandle(f(7))
        figure(7);
        uicontrol('Parent',f(7),'Style','pushbutton','Units','normalized','Position', [0.01 0.85 0.05 0.1],...
            'String', 'Date',...
            'Callback', {@startDateCallBack, f, files_list, prefixType, dataType, vascaType, anomalie});
        
        uicontrol('Parent',f(7),'Style','pushbutton','Units','normalized','Position', [0.01 0.70 0.05 0.10],...
            'String', strcat('ws ', num2str(window_size)),...
            'Callback', {@windowSizeCallBack, f, files_list, prefixType, dataType, vascaType, anomalie, date, window_size});
        
        for sensor_index=1:6
            subplot(2,3,sensor_index);
            hold on;
            grid on;
            grid minor;
            for v = 1:3
                curr_data = dati.vasca(v, sensor_index, :);
                curr_data = curr_data(:);
                Y = fft(curr_data);
                plot(abs(Y(1:numel(Y)/2)));
                xlabel(vascaType{sensor_index,1});
                ylabel('freq');
            end
            h = legend('vasca 1','vasca 2','vasca 3');
            rect = [0.0, 0.5, 0.1, 0.1];
            set(h, 'Position', rect);
            legend('boxoff');
        end
    end
    
    % plt pca
    if ishandle(f(8))
        figure(8);
        uicontrol('Parent',f(8),'Style','pushbutton','Units','normalized','Position', [0.01 0.85 0.05 0.1],...
            'String', 'Date',...
            'Callback', {@startDateCallBack, f, files_list, prefixType, dataType, vascaType, anomalie});
        
        uicontrol('Parent',f(8),'Style','pushbutton','Units','normalized','Position', [0.01 0.70 0.05 0.10],...
            'String', strcat('ws ', num2str(window_size)),...
            'Callback', {@windowSizeCallBack, f, files_list, prefixType, dataType, vascaType, anomalie, date, window_size});
        
        for v = 1:3
            subplot(1, 3, v);
            hold on;
            grid on;
            grid minor;
            % reshape data
            tmp_vasca = zeros(window_size, 6);
            for index = 1:window_size
                curr_measure = dati.vasca(v, :, index);
                curr_measure = curr_measure(:);
                tmp_vasca(index, :) = curr_measure;
            end
            [~, scores] = pca(tmp_vasca, 'NumComponents', 2);
            pca_X = scores(:, 1);
            pca_Y = scores(:, 2);
            plot(pca_X(:), pca_Y(:), 'o', 'color', colors(v));
            title(['PCA vasca', num2str(v)]);
        end
    end
    
    function startDateCallBack(~, ~, f, files_list, prefixType, dataType, vascaType, anomalie)
        new_start = inputdlg({'enter date or index'},'set starting date or index',1, {'dd/MM/yyyy HH'});
        if isempty(new_start) || new_start == ""
            return;
        end
        try
            new_date = datetime(new_start,'InputFormat','dd/MM/yyyy HH');
            plot2DProjection(f, files_list, prefixType, dataType, vascaType, anomalie, new_date);
        catch
            try
                new_index = str2double(new_start);
                if ~isempty(new_index) && new_index > 0 && ~isnan(new_index)
                    new_index = max(1, new_index);
                    new_index = min(new_index, size(files_list, 1));
                end
            catch
                warning('Can not parse date nor index');
                return;
            end
            plot2DProjection(f, files_list, prefixType, dataType, vascaType, anomalie, new_index);
        end
    end

    function windowSizeCallBack(~, ~, f, files_list, prefixType, dataType, vascaType, anomalie, date, window_size)
        new_window_size = inputdlg({'enter window size'},'set window size',1,{num2str(window_size)});
        
        if ~isempty(new_window_size)
            new_window_size = str2double(new_window_size);
            if new_window_size ~= window_size
                for i=1:numel(f)
                    if ishandle(f(i))
                        set(f(i), 'UserData', new_window_size);
                    end
                end
                plot2DProjection(f, files_list, prefixType, dataType, vascaType, anomalie, date);
            end
        end
    end
end
