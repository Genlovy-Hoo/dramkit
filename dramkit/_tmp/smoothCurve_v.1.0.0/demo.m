clear
clc
tic

% n = 80;
% x = 1:n;
% A = cos(2*pi*0.05*x+2*pi*rand) + 0.5*randn(1,n);
% B = smoothCurve(A);
% C = smooth(A);
% plot(x,A,'-o',x,B,'-x',x,C,'-*')
% legend('Original Data','Smoothed Data','matlab_smooth')


fpath = '../../test/510050_daily_pre_fq.csv';
data = csvread(fpath, 1, 2);

n = 200;
x = 1:n;
A = data(end-n+1:end, 5);
B = smoothCurve(A, 'w', 60, 'b', 20, 'method', 'konno-ohmachi');
C = smooth(A);
plot(x,A,'-o',x,B,'-x',x,C,'-*')
legend('Original Data','Smoothed Data','matlab_smooth')

toc
