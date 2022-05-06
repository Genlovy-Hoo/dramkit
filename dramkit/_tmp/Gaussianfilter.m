% 测试数据
x = 1:50;
y = x + rand(1,50)*10;

% 设置高斯模板大小和标准差
r = 3;
sigma = 2;
y_filted = Gaussianfilter1d(r, sigma, y);

% 作图对比
plot(x, y, '.-', x, y_filted);
title('高斯滤波');
legend('滤波前','滤波后','Location','northwest')

function y_filted = Gaussianfilter1d(r, sigma, y)

% 生成一维高斯滤波模板
GaussTemp = ones(1,r*2-1);
for i=1 : r*2-1
GaussTemp(i) = exp(-(i-r)^2/(2*sigma^2))/(sigma*sqrt(2*pi));
end

% 高斯滤波
y_filted = y;
for i = r : length(y)-r+1
y_filted(i) = y(i-r+1 : i+r-1)*GaussTemp';
end
end