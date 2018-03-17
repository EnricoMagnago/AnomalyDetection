%% Init the console
clear all;
close all;
clc;

addpath(genpath('Utility'));

%% load list of csv files, sort for ascending date.
if ~(exist('files_list.mat', 'file') == 2)
    cartelle = dir('../dataset/ETC_Riccione_*');
    list = {};
    for i=1:numel(cartelle)
        cartella = cartelle(i).folder;
        nome = cartelle(i).name;
        csv_files = [cartella,'/',nome,'/','*.csv'];
        subcartelle = dir(csv_files);
        for k=1:numel(subcartelle)
            csv_file = subcartelle(k);
            list = [list; {csv_file.name,[csv_file.folder,'/',csv_file.name]}];
        end
    end
    list = sort(list);
    files_list = {size(list,1), size(list, 2)};
    prev_file = '';
    for i=1:size(list,1)
        if strcmp(prev_file,list{i,1}) == 0
            file_time = datetime(regexp(list{i,1}, '\d*-\d*-\d*(-h\d*)?', 'match'),'InputFormat','yyyy-MM-dd-''h''HH');
            files_list{i,1} = file_time;
            files_list{i, 2} = list{i,2};
            prev_file = list{i,1};
        else
            error('duplicated file')
        end
    end

    save('files_list','files_list');
end

load('files_list');
%% Import the prefix map, init dati structure.
[ prefixType,dataType,vascaType, anomalie ] = mappaSigle( );

index = 1; % initial index
window_size = 60; % about 60 for each file.

%% Plot projection on each pair of sensors.
% portata,sst,ossigeno,ammoniaca,nitrati,valvola
f = [figure(1), figure(2), figure(3), figure(4), figure(5), figure(6), figure(7), figure(8)];

set(f, 'UserData', window_size);
plot2DProjection(f, files_list, prefixType, dataType, vascaType, anomalie, index);

%% Plot over time
% f_time = figure(7);
% set(f, 'UserData', window_size);
% plotOverTime(f_time, files_list, prefixType, dataType, vascaType, anomalie, index);
