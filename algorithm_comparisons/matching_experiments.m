clup

%% Problem
x = -2:0.01:2;
y = 1;
R = 1;

%% Make a graph
f0 = figure;
ylim([-6 0]);
hold on

f1 = figure;
% ylim([-6 0]);
hold on

f2 = figure;
% ylim([-6 0]);
hold on

%% True likelihood
h = x.^2;
lhood0 = log(mvnpdf(y, h', R));
% lhood1 = log(mvnpdf(y, h', R));
% lhood2 = log(mvnpdf(y, h', R));

figure(f0)
plot(x, lhood0);

%% Approximate
x0 = 0;

L1 = -(x0^2-1)*2*x0;
L2 = 2*(1-3*x0^2);

h0 = 1;
h1 = -L1;
h2 = -2*(L2+h1^2);

offset = log(mvnpdf(0, h0', R)) - log(mvnpdf(y, x0^2, R));
lapprox = log(mvnpdf(0, (h0+h1*(x-x0)+h2*(x-x0).^2)', R));
figure(f0)
plot(x, lapprox-offset, ':r');
