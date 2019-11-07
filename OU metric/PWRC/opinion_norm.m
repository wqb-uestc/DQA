function param = opinion_norm( scores, stds )
%OPINION_NORM Summary of this function goes here
%   MOS/DMOS normalization 
param.w = 1/(max(scores)-min(scores));
param.b = -min(scores)/(max(scores)-min(scores));
stds = stds*param.w*100;
param.th_min = 2*min(stds);
param.th_max = 2*max(stds);

end

