function [yhat, beta, ehat, rsqr] = regress_hamid(X, Y)

model = 'logistic_hamid';

temp = corrcoef(X,Y);

if (temp(1,2)>0)
    beta0(3) = mean(X);
    beta0(1) = abs(max(Y) - min(Y));
    beta0(4)=  mean(Y);
    beta0(2) = 1/std(X);
    beta0(5)=1;
else
    beta0(3) = mean(X);
    beta0(1) = - abs(max(Y) - min(Y));
    beta0(4)=  mean(Y);
    beta0(2) = 1/std(X);
    beta0(5)=1;
end

[beta ehat, J] = nlinfit(X, Y, model, beta0);
[yhat delta] = nlpredci(model, X, beta, ehat, J);

ehat = Y-yhat;
mY = mean(Y);
rsqr = 1 - sum(ehat.^2)/sum((Y-mY).^2);

return;