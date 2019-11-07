function score = delta_MOS( pred,label,param )
%TOPN_SCORE Summary of this function goes here
%   Detailed explanation goes here
%   MOS/DMOS normalization 
%   flag-1: MOS;  flag-0: DMOS
if param.flag
    label = (param.w*label + param.b)*100;
    pred  = (param.w*pred + param.b)*100;
else
    label = (1 - param.w*label - param.b)*100;
    pred  = (1 - param.w*pred - param.b)*100;
end
[~,idx] = sort(pred,'descend');
label = label(idx);
num = numel(label);
score = zeros(num-1,1);
for i = 1:num-1
    score(i) = mean(label(1:i))-mean(label(i+1:end));
end
score = mean(score);

end

