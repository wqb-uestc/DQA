function y = logistic_hamid(t, x)

a = t(1);
b = t(2);
c = t(3);
d = t(4);
e = t(5);

tmp = 0.5 - 1./(1+exp(b.*(x-c)));
y = a.*tmp + d + e.*x;

