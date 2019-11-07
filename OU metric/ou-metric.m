clc;clear;
close all;
model='ILNIQE';

addpath(genpath('./PWRC'));


load('..\test_id.txt');




all_id = 1:206;

for i = 1:10
    idx = ismember(all_id,test_id(i,:));
    ref_set_iter{i} = all_id(~idx);
    test_set_iter{i} = test_id(i,:);
end

load([model,'.mat']);
load('..\dataset\IVIPC_DQA\Data\realigned_mos.mat');
load('..\dataset\IVIPC_DQA\Data\realigned_std');
N = 10;
srocc_iter = zeros(N,1);
krocc_iter = zeros(N,1);
plcc_iter = zeros(N,1);

pscore=zeros(N,252);
tscore=zeros(N,252);
mosstd=zeros(N,252);

for iter = 1:N
    test_idx = test_set_iter{iter};
    test_mos = realigned_mos(test_idx,:);
    test_std=realigned_std(test_idx,:);
    test_mos = test_mos(:);
    test_std=test_std(:);
    pred_score = score(test_idx,:);
    pred_score = pred_score(:);
    srocc_iter(iter) = corr(pred_score,test_mos,'type','spearman');
    krocc_iter(iter) = corr(pred_score,test_mos,'type','kendall');
    plcc_iter(iter) = corr(pred_score,test_mos,'type','pearson');
    pscore(iter,:)=pred_score';
    tscore(iter,:)=test_mos';
    mosstd(iter,:)=test_std';
end
median_srocc = median(srocc_iter)
median_krocc = median(krocc_iter)
median_plcc = median(plcc_iter)


dlmwrite(['.\result\',model,'\pscore.txt'],pscore,'delimiter',' ');
dlmwrite(['.\result\',model,'\tscore.txt'],tscore,'delimiter',' ');
dlmwrite(['.\result\',model,'\std.txt'],mosstd,'delimiter',' ');



for shuju =1:10
    
    
    delimiterIn   = ' '; 
    headerlinesIn = 0;   
    A = importdata(['.\result\',model,'\tscore.txt'], delimiterIn, headerlinesIn);
    B = importdata(['.\result\',model,'\pscore.txt'], delimiterIn, headerlinesIn);
    dmos = A(shuju,1:end)' ;
    LIVE_ssim =B(shuju,1:end)' ;
    C = importdata(['.\result\',model,'\std.txt'], delimiterIn, headerlinesIn);
    dmos_std=C(shuju,:);
    pred = regress_hamid(LIVE_ssim,dmos);
    % parameter preparation
    p = opinion_norm(dmos,dmos_std);
    p.flag = 0;     % DMOS -> flag 0; MOS -> flag 1;
    p.act  = 1;     % enable/disable A(x,T): p.act->1/0 
    th = 0:0.5:110; % customize observation interval

    [PWRC_th,AUC] = PWRC(pred,dmos,th,p); 
    disp(['The AUC value of SSIM is ',num2str(AUC)]);

    delta_value = delta_MOS(pred,dmos,p);
    disp(['The delta MOS value of SSIM is ',num2str(delta_value)]);
    
    AUCall(shuju)=AUC ;
    delta_valueall(shuju)=delta_value;
    PWRCall(shuju,:)=PWRC_th ;
    
end
AUCmedia = median(AUCall)
resultmedia=[median_srocc,median_plcc,median_krocc,AUCmedia]
dlmwrite(['.\result\',model,'\testresult.txt'],resultmedia,'delimiter',' ');
