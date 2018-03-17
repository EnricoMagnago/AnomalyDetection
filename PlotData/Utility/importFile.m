function [ dati, is_nan_list ] = importFile( fileName,prefixType )
%IMPORTFILE Summary of this function goes here
%   Detailed explanation goes here


% Init data struct
dati.time = [datetime,datetime];
dati.vasca = zeros(3,5,0);
dati.soffiatore = zeros(6,0);

is_nan_list = [];

%% Import data
fid = fopen(fileName,'r');

time_indx = 0;
while true
    tline = fgetl(fid);    
    if not(ischar(tline))
       break; 
    end
    
    line_splt = split(tline,';');
    timestamp = datetime(line_splt(1),'InputFormat','dd/MM/yyyy HH:mm:ss');
    if time_indx == 0 || not(dati.time(time_indx)==timestamp)
        time_indx = time_indx + 1;
        dati.time(time_indx) = timestamp;
        dati.vasca(:,:,time_indx) = -1;
        dati.soffiatore(:,time_indx) = -1; % valori non inizializzati sono -1
    end
    
    for i=1:numel(prefixType)
        if strcmp(line_splt{2},prefixType{i}{1}) % find line that refers to that sensor.
            if prefixType{i}{2}==1 % vasca
                indx = prefixType{i}{3}; % list of 2: line, sensor_type.
                % dati.vasca(line, sensor_type, time) = measurement
                measurement = str2num(line_splt{3});
                if isempty(measurement) || measurement < 0 || measurement > 500
                    measurement = nan;
                    is_nan_list = [is_nan_list; true];
                else
                    is_nan_list = [is_nan_list; false];
                end
                dati.vasca(indx(1),indx(2),time_indx) = measurement; 
            elseif prefixType{i}{2}==2 % soffiatore
                indx = prefixType{i}{3};
                % dati.vasca( soff_id, time) = measurement
                dati.vasca(indx(1),time_indx) = str2num(line_splt{3});
            else
                error('unknown measurement type');
            end
            break; % found and stored, go to the next line.
        end
    end
end

is_nan_list = logical(is_nan_list);

fclose(fid);

end

