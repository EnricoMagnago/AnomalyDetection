function [ prefixType,dataType,vascaType, anomalie ] = mappaSigle( )
%MAPPASIGLE Summary of this function goes here
%   Detailed explanation goes here
prefixType = {};
index = 0;
dataType = {'vasca',1;'soffiatore',2};
vascaType = {'portata',1;'sst',2;'ossigeno',3;'ammoniaca',4;'nitrati',5;'valvola',6};
% SO01_01-aMisura: Portata Linea 1 [mc/h]
index = index + 1;
prefixType{index}{1} = 'SO01_01-aMisura';
prefixType{index}{2} = dataType{1,2}; % vasca
prefixType{index}{3} = [1,vascaType{1,2}]; % linea_1, portata
% SO01_02-aMisura : Portata Linea 2 [mc/h]
index = index + 1;
prefixType{index}{1} = 'SO01_02-aMisura';
prefixType{index}{2} = dataType{1,2}; % vasca
prefixType{index}{3} = [2,vascaType{1,2}]; % linea_2, portata
% SO01_03-aMisura: Portata Linea 3 [mc/h]
index = index + 1;
prefixType{index}{1} = 'SO01_03-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [3,vascaType{1,2}];
% SO01_04-aMisura: SST (solidi sospesi) Linea 1 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO01_04-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [1,vascaType{2,2}];
% SO01_05-aMisura: SST (solidi sospesi) Linea 2 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO01_05-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [2,vascaType{2,2}];
% SO01_06-aMisura: SST (solidi sospesi) Linea 3 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO01_06-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [3,vascaType{2,2}];
% SO03_01-aMisura: Ossigeno Linea 1 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO03_01-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [1,vascaType{3,2}];
% SO05_01-aMisura: Ossigeno Linea 2 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO05_01-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [2,vascaType{3,2}];
% SO07_01-aMisura: Ossigeno Linea 3 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO07_01-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [3,vascaType{3,2}];
% SO03_04-aMisura: Ammoniaca Linea 1 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO03_04-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [1,vascaType{4,2}];
% SO05_04-aMisura: Ammoniaca Linea 2 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO05_04-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [2,vascaType{4,2}];
% SO07_04-aMisura: Ammoniaca Linea 3 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO07_04-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [3,vascaType{4,2}];
% SO03_05-aMisura: Nitrati Linea 1 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO03_05-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [1,vascaType{5,2}];
% SO05_05-aMisura: Nitrati Linea 2 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO05_05-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [2,vascaType{5,2}];
% SO07_05-aMisura: Nitrati Linea 3 [mg/L]
index = index + 1;
prefixType{index}{1} = 'SO07_05-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [3,vascaType{5,2}];
% CR03_01-aAssorbimento: Potenza soffiante 1 [kW]
sof_num = 0;
index = index + 1;
sof_num = sof_num + 1;
prefixType{index}{1} = 'CR03_01-aAssorbimento';
prefixType{index}{2} = dataType{2,2};
prefixType{index}{3} = sof_num;
% CR03_02-aAssorbimento: Potenza soffiante 2 [kW]
index = index + 1;
sof_num = sof_num + 1;
prefixType{index}{1} = 'CR03_02-aAssorbimento';
prefixType{index}{2} = dataType{2,2};
prefixType{index}{3} = sof_num;
% CR03_03-aAssorbimento: Potenza soffiante 3 [kW]
index = index + 1;
sof_num = sof_num + 1;
prefixType{index}{1} = 'CR03_03-aAssorbimento';
prefixType{index}{2} = dataType{2,2};
prefixType{index}{3} = sof_num;
% CR03_04-aAssorbimento: Potenza soffiante 4 [kW]
index = index + 1;
sof_num = sof_num + 1;
prefixType{index}{1} = 'CR03_04-aAssorbimento';
prefixType{index}{2} = dataType{2,2};
prefixType{index}{3} = sof_num;
% CR03_05-aAssorbimento: Potenza soffiante 5 [kW]
index = index + 1;
sof_num = sof_num + 1;
prefixType{index}{1} = 'CR03_05-aAssorbimento';
prefixType{index}{2} = dataType{2,2};
prefixType{index}{3} = sof_num;
% CR03_06-aAssorbimento: Potenza soffiante 6 [kW]
index = index + 1;
sof_num = sof_num + 1;
prefixType{index}{1} = 'CR03_06-aAssorbimento';
prefixType{index}{2} = dataType{2,2};
prefixType{index}{3} = sof_num;
% SO03_07-aMisura: Valvola Linea 1 (apertura) [%]
index = index + 1;
prefixType{index}{1} = 'SO03_07-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [1,vascaType{6,2}];
% SO05_07-aMisura: Valvola Linea 2 (apertura) [%]
index = index + 1;
prefixType{index}{1} = 'SO05_07-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [2,vascaType{6,2}];
% SO07_07-aMisura: Valvola Linea 3 (apertura) [%]
index = index + 1;
prefixType{index}{1} = 'SO07_07-aMisura';
prefixType{index}{2} = dataType{1,2};
prefixType{index}{3} = [3,vascaType{6,2}];

% blu : portata_sst
% arancio : ammoniaca
% verde: ossigeno
% rosa: macchine
% list: description, lines, {begin, end}

anomalie = [
            {'valvole', [2], {datetime('19/04/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('20/04/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'portata_sst_diluito', [1,2,3], {datetime('11/05/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('12/05/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'valvole', [1,2,3], {datetime('23/05/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('24/05/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca_ossigeno_nitrati', [1,2,3], {datetime('26/05/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('27/05/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'avviamento', [3], {datetime('30/05/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('30/05/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'nitrati_wrong', [3], {datetime('30/05/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('31/05/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'nitrati_deriva', [3], {datetime('01/06/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('07/06/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'ammoniaca_wrong', [1,2,3], {datetime('02/06/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('09/06/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca_ossigeno_nitrati', [3], {datetime('30/06/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('01/07/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'controllore_rotto', [1,2,3], {datetime('01/07/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('04/07/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'poca_aria', [1,2,3], {datetime('22/07/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('25/07/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca_nitrati', [3], {datetime('02/08/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('03/08/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca', [1,2,3], {datetime('17/08/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('24/08/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca', [3], {datetime('22/09/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('23/09/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_nitrati', [3], {datetime('23/09/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('24/09/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca_nitrati', [1, 2], {datetime('06/10/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('07/10/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca', [3], {datetime('27/10/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('28/10/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammonica', [1, 2, 3], {datetime('09/11/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('10/11/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_nitrati', [1, 2], {datetime('09/11/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('10/11/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'sminchiate_soglie_ammoniaca', [2], {datetime('17/11/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('29/11/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'sminchiate_soglie_ossigeno', [3], {datetime('17/11/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('29/11/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'disabilitazione_controllore', [1,2,3], {datetime('02/12/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('06/12/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'sst', [1, 2, 3], {datetime('11/12/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('15/12/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'ossigeno', [1, 2, 3], {datetime('21/12/2016 00','InputFormat','dd/MM/yyyy HH'), datetime('24/12/2016 00','InputFormat','dd/MM/yyyy HH')}};...
            {'deriva_ossigeno', [1, 2, 3], {datetime('02/01/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('08/02/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca_nitrati', [3], {datetime('08/02/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('09/02/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca', [3], {datetime('20/02/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('21/02/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca', [1, 2, 3], {datetime('16/03/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('22/03/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca_nitrati', [1, 2, 3], {datetime('27/03/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('28/03/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'wrong_ammoniaca', [2], {datetime('03/04/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('11/04/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_nitrati', [3], {datetime('03/04/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('11/04/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ossigeno', [1, 2, 3], {datetime('04/04/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('06/04/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'soffianti', [1, 2, 3], {datetime('23/05/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('24/05/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'deriva_nitrati', [1, 2, 3], {datetime('23/05/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('30/05/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'deriva_ammoniaca', [1, 2, 3], {datetime('25/05/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('07/06/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_ammoniaca', [3], {datetime('07/06/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('08/06/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'calibrazione_nitrati', [1, 2], {datetime('07/06/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('08/06/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'sst_ammoniaca', [1, 2, 3] {datetime('09/07/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('12/07/2017 00','InputFormat','dd/MM/yyyy HH')}};...
            {'wrong_ammoniaca', [1], {datetime('12/07/2017 00','InputFormat','dd/MM/yyyy HH'), datetime('14/07/2017 00','InputFormat','dd/MM/yyyy HH')}}...
           ];
end
