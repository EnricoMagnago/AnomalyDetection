function plotOverTime(f, files_list, prefixType, dataType, vascaType, anomalie, date)
    window_size = get(f, 'UserData');
    clf(f);
    
    dati = loadDataFromDateOrIndex(files_list, date, window_size, prefixType);
    window_size = size(dati.time, 2);
    current_anomalies = filter_anomalies_by_date(anomalie, dati.time(1), dati.time(size(dati.time, 2)));

    set(f, 'UserData', window_size);
    
    uicontrol('Parent',f,'Style','pushbutton','Units','normalized','Position', [0.01 0.85 0.05 0.1],...
              'String', 'Date',...
              'Callback', {@startDateCallBack, f, files_list, prefixType, dataType, vascaType, anomalie});
    
          
    uicontrol('Parent',f,'Style','pushbutton','Units','normalized','Position', [0.01 0.70 0.05 0.10],...
              'String', strcat('ws ', num2str(window_size)),...
              'Callback', {@windowSizeCallBack, f, files_list, prefixType, dataType, vascaType, anomalie, date, window_size});
  
    set(f, 'NumberTitle', 'off', 'Name', sprintf('%d: from %s to %s',...
        size(dati.time, 2), datestr(dati.time(1)), datestr(dati.time(size(dati.time, 2)))));
    colors = ['r', 'g', 'b'];
    width = [1.2, 0.8, 0.4];
    plots = zeros(3,1);
    for i=1:6
        subplot(2,3,i);
        hold on;
        title(vascaType{i,1});
        for v = 1:3
            plots(v) = plot(dati.time,reshape(dati.vasca(v,i,:), 1, numel(dati.time)), colors(v), 'LineWidth', width(v));
        end
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
    end
    
    legend(plots, 'vasca 1','vasca 2','vasca 3', 'Position', [15, 10 + 20, 10, 10]);
    legend('boxoff');
    
    function startDateCallBack(h, ~, f, files_list, prefixType, dataType, vascaType, anomalie)
        new_start = inputdlg({'enter date or index'},'set starting date or index',1, {'dd/MM/yyyy HH'});
        try
            new_date = datetime(new_start,'InputFormat','dd/MM/yyyy HH');
            plotOverTime(f, files_list, prefixType, dataType, vascaType, anomalie, new_date);
        catch
            try
                new_index = str2double(new_start);
                if ~isempty(new_index)
                    new_index = max(1, new_index);
                    new_index = min(new_index, size(files_list, 1));
                    plotOverTime(f, files_list, prefixType, dataType, vascaType, anomalie, new_index);
                end
            catch
                warning('Can not parse date nor index');
            end
        end
    end

    function windowSizeCallBack(~, ~, f, files_list, prefixType, dataType, vascaType, anomalie, date, window_size)
        new_window_size = inputdlg({'enter window size'},'set window size',1,{num2str(window_size)});
        
        if ~isempty(new_window_size)
            new_window_size = str2double(new_window_size);
            if new_window_size ~= window_size        
                set(f, 'UserData', new_window_size);
                plotOverTime(f, files_list, prefixType, dataType, vascaType, anomalie, date);
            end
        end
    end
end