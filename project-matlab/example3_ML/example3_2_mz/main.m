clear, clc;
%%
close all;
%%
addpath("aerodynamic\");

%% Формирование обучающего множества

alpha = deg2rad(linspace(-20, 90, 100));
beta = deg2rad(0);
phi = deg2rad(-25:0.5:25);
dnos = 0;
omega_z = 0;
V = 150; % м/с
ba = 3.45; % м
sb = 0;

get_mz = @(alpha, phi) GetMz(alpha, beta, phi, dnos, omega_z, V, ba, sb);

for i = 1:length(phi)
    for j = 1:length(alpha)
        mz(i, j) = get_mz(alpha(j), phi(i));
    end
end

% Создание сетки
[x, y] = meshgrid(alpha, phi);

inputs = [x(:)'; y(:)'];
targets = mz(:)';

% Нормализация данных
[inputs_norm, input_ps] = mapminmax(inputs, -1, 1);
[targets_norm, target_ps] = mapminmax(targets, -1, 1);

%% Создание нейронной сети

net = fitnet([10, 15, 20], 'trainlm');

% Настройка параметров
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 25;
net.trainParam.min_grad = 1e-6;
net.performFcn = 'mse';

% Настройка разделения данных
net.divideParam.trainRatio = 0.85;   % 85% на обучение
net.divideParam.valRatio = 0.10;     % 10% на валидацию
net.divideParam.testRatio = 0.05;    % 5% на тестирование

% Настройка функций активации
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';

%% Обучение сети
[net, tr] = train(net, inputs_norm, targets_norm);

%% Проверка точности
% Предсказание на нормализованных данных
outputs_norm = net(inputs_norm);

% Денормализация результатов
outputs = mapminmax('reverse', outputs_norm, target_ps);

% Расчет ошибки
abs_error = abs(outputs - targets);
mse_error = mean((outputs - targets).^2);
max_error = max(abs_error);

fprintf('Среднеквадратичная ошибка (MSE): %.6f\n', mse_error);
fprintf('Максимальная абсолютная ошибка: %.6f\n', max_error);
fprintf('Средняя абсолютная ошибка: %.6f\n', mean(abs_error));

mz_net = reshape(outputs, size(x));

%% Визуализация результатов

indx_phi = find(phi == deg2rad(-25) | phi == deg2rad(-12.5) | phi == 0 | phi == deg2rad(12.5) | phi == deg2rad(25));

% исходная зависимость
figure;
surf(rad2deg(x), rad2deg(y), mz, 'EdgeColor', 'none');
title('Исходная зависимость m_z');
xlabel('$\alpha$, $\circ$', 'Interpreter', 'latex');
ylabel('$\varphi$, $\circ$', 'Interpreter', 'latex');
zlabel('$m_{z}$($\alpha$, $\varphi$)', 'Interpreter', 'latex');
colorbar;

figure;
hold on;
grid on;
for i = 1:length(indx_phi)
    plot(rad2deg(alpha), mz(indx_phi(i), :));
    legs{i} = "$\varphi$ = " + num2str(rad2deg(phi(indx_phi(i)))) + "$^\circ$";
end
title('Исходная зависимость m_z');
ylabel('$m_{z}$($\alpha$, $\varphi$)', 'Interpreter', 'latex');
xlabel('$\alpha$, $\circ$', 'Interpreter','latex');
legend(legs, 'Interpreter','latex');

% аппроксимация нейросетью
figure;
surf(rad2deg(x), rad2deg(y), mz_net, 'EdgeColor', 'none');
title('Аппроксимация нейросетью зависимости m_z');
xlabel('$\alpha$, $\circ$', 'Interpreter', 'latex');
ylabel('$\varphi$, $\circ$', 'Interpreter', 'latex');
zlabel('$m_{z}$($\alpha$, $\varphi$)', 'Interpreter', 'latex');
colorbar;

figure;
hold on;
grid on;
for i = 1:length(indx_phi)
    plot(rad2deg(alpha), mz_net(indx_phi(i), :));
end
title('Аппроксимация нейросетью зависимости m_z');
ylabel('$m_{z}$($\alpha$, $\varphi$)', 'Interpreter', 'latex');
xlabel('$\alpha$, $\circ$', 'Interpreter','latex');
legend(legs, 'Interpreter','latex');

% ошибка аппроксимации
abs_error_mat = reshape(abs_error, size(x));

figure;
surf(rad2deg(x), rad2deg(y), abs_error_mat, 'EdgeColor', 'none');
title('Абсолютная ошибка аппроксимации m_z');
xlabel('$\alpha$, $\circ$', 'Interpreter', 'latex');
ylabel('$\varphi$, $\circ$', 'Interpreter', 'latex');
zlabel('$E_{m_{z}}$', 'Interpreter', 'latex');
colorbar;

figure;
hold on;
grid on;
for i = 1:length(indx_phi)
    plot(rad2deg(alpha), abs_error_mat(indx_phi(i), :));
end
title('Абсолютная ошибка аппроксимации m_z');
ylabel('$E_{m_{z}}$', 'Interpreter', 'latex');
xlabel('$\alpha$, $\circ$', 'Interpreter', 'latex');
legend(legs, 'Interpreter', 'latex');

%% Дифференцирование методом конечных разностей

eps = 1e-4;

% вычисление частной производной по углу атаки
for i = 1:length(phi)
    for j = 1:length(alpha)
        dmz_dalpha_plus = get_mz(alpha(j) + eps, phi(i));
        dmz_dalpha_minus = get_mz(alpha(j) - eps, phi(i));
        dmz_dalpha(i, j) = (dmz_dalpha_plus - dmz_dalpha_minus)/(2*eps);
    end
end

% вычисление частной производной по углу отклонения стабилизатора
for i = 1:length(phi)
    for j = 1:length(alpha)
        dmz_dphi_plus = get_mz(alpha(j), phi(i) + eps);
        dmz_dphi_minus = get_mz(alpha(j), phi(i) - eps);
        dmz_dphi(i, j) = (dmz_dphi_plus - dmz_dphi_minus)/(2*eps);
    end
end

%% Алгоритмическое дифференцирование нейросети

% весовые коэффициенты и смещения
W1 = net.IW{1, 1};
b1 = net.b{1};
W2 = net.LW{2, 1};
b2 = net.b{2};
W3 = net.LW{3, 2};
b3 = net.b{3};
W4 = net.LW{4, 3};
b4 = net.b{4};

% параметры нормализации
in_min = input_ps.xmin;    % [x_min; y_min]
in_max = input_ps.xmax;    % [x_max; y_max]
out_min = target_ps.xmin;  % Минимум z
out_max = target_ps.xmax;  % Максимум z

% создание функции для вычисления выхода сети и его частных производных по
% входам
neural_net_function = @(x, y) func(x, y, ...
                        W1, b1, W2, b2, W3, b3, W4, b4, ...
                        in_min, in_max, out_min, out_max);

dmz_dalpha_net = zeros(size(x));
dmz_dphi_net = zeros(size(x));

for i = 1:numel(x)
    [~, grads] = dlfeval(neural_net_function, dlarray(x(i)), dlarray(y(i)));
    dmz_dalpha_net(i) = extractdata(grads{1});
    dmz_dphi_net(i) = extractdata(grads{2});
end

%% Визуализация

figure;
subplot(2, 1, 1);
surf(rad2deg(x), rad2deg(y), dmz_dalpha, 'EdgeColor', 'none');
xlabel('$\alpha$, $\circ$', 'Interpreter', 'latex');
ylabel('$\varphi$, $\circ$', 'Interpreter', 'latex');
zlabel('$m_{z}^\alpha$, 1/rad', 'Interpreter', 'latex');
title(["Частные производные исходной зависимости m_z,"; "полученные методом конечных разностей"]);
colorbar;

subplot(2, 1, 2);
surf(rad2deg(x), rad2deg(y), dmz_dphi, 'EdgeColor', 'none');
xlabel('$\alpha$, $\circ$', 'Interpreter', 'latex');
ylabel('$\varphi$, $\circ$', 'Interpreter', 'latex');
zlabel('$m_{z}^\varphi$, 1/rad', 'Interpreter', 'latex');
colorbar;

figure;
subplot(2, 1, 1);
surf(rad2deg(x), rad2deg(y), dmz_dalpha_net, 'EdgeColor', 'none');
xlabel('$\alpha$, $\circ$', 'Interpreter', 'latex');
ylabel('$\varphi$, $\circ$', 'Interpreter', 'latex');
zlabel('$m_{z}^\alpha$, 1/rad', 'Interpreter', 'latex');
title(["Частные производные нейросетевой аппроксимации m_z,"; "полученные методом АД"]);
colorbar;

subplot(2, 1, 2);
surf(rad2deg(x), rad2deg(y), dmz_dphi_net, 'EdgeColor', 'none');
xlabel('$\alpha$, $\circ$', 'Interpreter', 'latex');
ylabel('$\varphi$, $\circ$', 'Interpreter', 'latex');
zlabel('$E_{m_{z}^\varphi}$, 1/rad', 'Interpreter', 'latex');
colorbar;

figure;
subplot(2, 1, 1);
surf(rad2deg(x), rad2deg(y), abs(dmz_dalpha - dmz_dalpha_net), 'EdgeColor', 'none');
xlabel('$\alpha$, $\circ$', 'Interpreter', 'latex');
ylabel('$\varphi$, $\circ$', 'Interpreter', 'latex');
zlabel('$E_{m_{z}^\alpha}$, 1/rad', 'Interpreter', 'latex');
title(["Абсолютная разность значений частных производных,"; "полученных методами конечных разностей и АД"]);
colorbar;

subplot(2, 1, 2);
surf(rad2deg(x), rad2deg(y), abs(dmz_dphi - dmz_dphi_net), 'EdgeColor', 'none');
xlabel('$\alpha$, $\circ$', 'Interpreter', 'latex');
ylabel('$\varphi$, $\circ$', 'Interpreter', 'latex');
zlabel('$m_{z}^\varphi$, 1/rad', 'Interpreter', 'latex');
colorbar;