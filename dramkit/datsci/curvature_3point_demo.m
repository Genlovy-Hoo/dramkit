clear
clc
tic

theta = 0:0.2:2*pi;

kappa_arr = [];
posi_arr = [];
norm_arr = [];

for num = 2: (length(theta)-1)
    x = 5 * sin(theta(num-1: num+1));
    y = 5 * cos(theta(num-1: num+1));
    [curvature, dircts] = Curvature_3Point(x, y);
    posi_arr = [posi_arr; [x(2), y(2)]];
    kappa_arr = [kappa_arr; curvature];
    norm_arr = [norm_arr; dircts];
end


quiver(posi_arr(:, 1), posi_arr(:, 2),...
       kappa_arr .* norm_arr(:, 1), kappa_arr .* norm_arr(:, 2))
   
toc
   
%%
function [curvature, dircts] = Curvature_3Point(x, y)
    
    % 根据三个离散点计算曲率
    %     
    % Parameters
    % ----------
    % x: 三个点x轴坐标列表
    % y: 三个点y轴坐标列表
    %     
    % Returns
    % -------
    % curvature: 曲率大小
    % dircts: 曲率方向（标准化）
    %     
    % 参考：
    % https://github.com/Pjer-zhang/Curvature_3Point
    % https://zhuanlan.zhihu.com/p/72083902
    
    x = reshape(x, 3, 1);
    y = reshape(y, 3, 1);
    t_a = norm([x(2)-x(1), y(2)-y(1)]);
    t_b = norm([x(3)-x(2), y(3)-y(2)]);
    
    M =[1, -t_a, t_a^2; 1, 0, 0; 1, t_b, t_b^2];

    a = M\x;
    b = M\y;

    curvature = 2 * (a(3)*b(2) - b(3)*a(2)) / (a(2)^2 + b(2)^2)^(1.5);
    dircts = [b(2), -a(2)] / sqrt(a(2)^2 + b(2)^2);
    
end

