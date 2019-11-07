function [ er_rate, AUC ] = PWRC( pred, label, th, param )
%UIRM Summary of this function goes here
%   check whether inputs are column vector or not
if size(pred,1)==1
    pred = pred';
end
if size(label,1)==1
    label = label';
end
%   MOS/DMOS normalization 
%   flag-1: MOS;  flag-0: DMOS
if param.flag
    label = (param.w*label + param.b)*100;
    pred  = (param.w*pred + param.b)*100;
else
    label = (1 - param.w*label - param.b)*100;
    pred  = (1 - param.w*pred - param.b)*100;
end
%   locate range index
[~,min_idx] = min(abs(th - param.th_min));
[~,max_idx] = min(abs(th - param.th_max));

c = 0.175;
%   re-rank scores
[~,~,label_r] = unique(label);
[~,~,pred_r] = unique(pred);

len = numel(label);
len_short = 2*(len-1);
%   pre-computing for importance weight w
w = cell(len-1,1);
omega = 0;
for i = 1:len-1
    level = max([label_r(1:end-i),label_r(i+1:end)],[],2)-1;
    diff = abs(label_r(1:end-i)-pred_r(1:end-i))+...
        abs(label_r(i+1:end)-pred_r(i+1:end));
    w{i} = exp(level/(len-1)+diff/len_short);
    omega = omega + sum(w{i});
end
for i = 1:len-1
    w{i} = w{i}/omega;
end

if param.act
    er_rate = zeros(1,numel(th));
    for step = 1:numel(th)   
        for i = 1:len-1
            temp_pred  = pred(1:end-i)-pred(i+1:end);
            temp_label = label(1:end-i)-label(i+1:end);
            active_label = 1./(1+exp(-c*(abs(temp_label)-th(step))));
            er_label = sign(temp_pred).*sign(temp_label);        
            idx = er_label.*active_label;
            er_rate(step) = er_rate(step) + sum(w{i}.*idx);        
        end
    end
    AUC = trapz(th(min_idx:max_idx),er_rate(min_idx:max_idx));
else
    er_rate = 0;
    for i = 1:len-1
        temp_pred  = pred(1:end-i)-pred(i+1:end);
        temp_label = label(1:end-i)-label(i+1:end);
        er_label = sign(temp_pred).*sign(temp_label);        
        er_rate = er_rate + sum(w{i}.*er_label);        
    end
    AUC = er_rate;
end

end

