clear, clc;
close all;
%% 1. Генерация данных
x_min = -2; x_max = 2;
y_min = -2; y_max = 2;
step = 0.15;

% Создание сетки
[x, y] = meshgrid(x_min:step:x_max, y_min:step:y_max);
z = sin(x.^2 + y.^2);

inputs = [x(:)'; y(:)'];  
targets = z(:)';          

%% 2. Нормализация данных
[inputs_norm, input_ps] = mapminmax(inputs, -1, 1); % Нормализация входов [-1, 1]
[targets_norm, target_ps] = mapminmax(targets, -1, 1); % Нормализация выходов

%% 3. Создание нейронной сети
net = fitnet(45, 'trainlm');

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
net.layers{1}.transferFcn = 'tansig'; % Гиперболический тангенс (1 слой)

%% 4. Обучение сети
[net, tr] = train(net, inputs_norm, targets_norm);

%% 5. Проверка точности
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

%% 6. Визуализация результатов
% исходная функция
figure('Position', [100 100 1200 500]);
subplot(1,3,1);
surf(x, y, z, 'EdgeColor', 'none');
title('Исходная функция: sin(x^2 + y^2)');
xlabel('x'); ylabel('y'); zlabel('z');
axis tight;
colorbar;

% аппроксимация нейросетью
subplot(1,3,2);
surf(x, y, reshape(outputs, size(x)), 'EdgeColor', 'none');
title('Аппроксимация нейросетью');
xlabel('x'); ylabel('y');
axis tight;
colorbar;

% ошибка аппроксимации
subplot(1,3,3);
surf(x, y, reshape(abs_error, size(x)), 'EdgeColor', 'none');
title('Абсолютная ошибка');
xlabel('x'); ylabel('y');
axis tight;
colorbar;
colormap jet;

%% 7. Пример использования
test_points = [
    -1.5, 0.8;
    0.3, -1.1;
    1.7, 1.9;
    0, 0;
    -0.7, -1.8
];

fprintf('\nТестирование в точках:\n');
fprintf('   x      y     |  Истинное  Предсказанное  Ошибка\n');
fprintf('----------------|-------------------------------\n');

for i = 1:size(test_points, 1)
    pt = test_points(i,:)';
    true_val = sin(pt(1)^2 + pt(2)^2);
    
    % нормализация входов
    pt_norm = mapminmax('apply', pt, input_ps);
    
    % предсказание
    pred_norm = net(pt_norm);
    pred = mapminmax('reverse', pred_norm, target_ps);
    
    err = abs(true_val - pred);
    fprintf('%6.2f  %6.2f | %9.6f  %9.6f  %9.6f\n', ...
            pt(1), pt(2), true_val, pred, err);
end

%% 8. Извлечение параметров сети
% весовые коэффициенты и смещения
W1 = net.IW{1,1};
b1 = net.b{1};
W2 = net.LW{2,1};
b2 = net.b{2};

% параметры нормализации
in_min = input_ps.xmin;    % [x_min; y_min]
in_max = input_ps.xmax;    % [x_max; y_max]
out_min = target_ps.xmin;  % Минимум z
out_max = target_ps.xmax;  % Максимум z


%% 9. Проверка формулы нейросети на тестовой точке

test_x = -1.5; test_y = 0.8;

% нормализация входа
x_norm = 2*(test_x - in_min(1))/(in_max(1) - in_min(1)) - 1;
y_norm = 2*(test_y - in_min(2))/(in_max(2) - in_min(2)) - 1;
input_norm = [x_norm; y_norm];

L1 = tanh( W1 * input_norm + b1 );
z_norm = W2 * L1 + b2;

% денормализация
z_pred = (z_norm + 1)*(out_max - out_min)/2 + out_min;

% сравнение с сетью
pt_norm = mapminmax('apply', [test_x; test_y], input_ps);
net_pred = mapminmax('reverse', net(pt_norm), target_ps);

fprintf('\nПроверка формулы на точке (%.1f, %.1f):\n', test_x, test_y);
fprintf('Расчет по формуле: z = %.6f\n', z_pred);
fprintf('Расчет сетью:     z = %.6f\n', net_pred);
fprintf('Разница: %.12f\n', abs(z_pred - net_pred));

%% 11. Алгоритмическое дифференцирование нейросети
% создание функции для вычисления выхода сети и его производных
neural_net_function = @(x, y) func(x, y, W1, b1, W2, b2, ...
                                                     in_min, in_max, out_min, out_max);

% тестовая точка
test_x = -1.5;
test_y = 0.8;

% вычисление с помощью AD
[net_value, net_grad] = dlfeval(neural_net_function, dlarray(test_x), dlarray(test_y));

% извлечение результатов
net_value = extractdata(net_value);
net_grad_x = extractdata(net_grad{1});
net_grad_y = extractdata(net_grad{2});

%% 10. Аналитическое дифференцирование исходной функции
% исходная функция: f = sin(x² + y²)
syms x_sym y_sym;
f_analytic = sin(x_sym^2 + y_sym^2);

% Аналитические производные
df_dx_analytic = diff(f_analytic, x_sym);
df_dy_analytic = diff(f_analytic, y_sym);

% Вычисление в тестовой точке
analytic_value = double(subs(f_analytic, [x_sym, y_sym], [test_x, test_y]));
analytic_grad_x = double(subs(df_dx_analytic, [x_sym, y_sym], [test_x, test_y]));
analytic_grad_y = double(subs(df_dy_analytic, [x_sym, y_sym], [test_x, test_y]));

%% 11. Сравнение результатов
fprintf('\nСравнение в точке (%.1f, %.1f):\n', test_x, test_y);
fprintf('%-25s | %-15s | %-15s\n', 'Метрика', 'Нейросеть', 'Аналитика');
fprintf('-------------------------|----------------|----------------\n');
fprintf('%-25s | %-14.6f | %-14.6f\n', 'Значение функции', net_value, analytic_value);
fprintf('%-25s | %-14.6f | %-14.6f\n', '∂f/∂x', net_grad_x, analytic_grad_x);
fprintf('%-25s | %-14.6f | %-14.6f\n', '∂f/∂y', net_grad_y, analytic_grad_y);
fprintf('\nАбсолютные ошибки:\n');
fprintf('Δf:    %.6e\n', abs(net_value - analytic_value));
fprintf('Δ∂f/∂x: %.6e\n', abs(net_grad_x - analytic_grad_x));
fprintf('Δ∂f/∂y: %.6e\n', abs(net_grad_y - analytic_grad_y));

%% 12. Визуализация производных

[x_vis, y_vis] = meshgrid(-2:0.2:2, -2:0.2:2);

df_dx_vis = 2*x_vis.*cos(x_vis.^2 + y_vis.^2);
df_dy_vis = 2*y_vis.*cos(x_vis.^2 + y_vis.^2);

df_dx_net = zeros(size(x_vis));
df_dy_net = zeros(size(x_vis));

for i = 1:numel(x_vis)
    [~, grads] = dlfeval(neural_net_function, dlarray(x_vis(i)), dlarray(y_vis(i)));
    df_dx_net(i) = extractdata(grads{1});
    df_dy_net(i) = extractdata(grads{2});
end

figure('Position', [100 100 1200 800]);

% ∂f/∂x аналитический
subplot(2,2,1);
surf(x_vis, y_vis, df_dx_vis, 'EdgeColor', 'none');
title('Аналитическая ∂f/∂x');
xlabel('x'); ylabel('y'); zlabel('∂f/∂x');
colorbar;
colormap jet;

% ∂f/∂x нейросеть
subplot(2,2,2);
surf(x_vis, y_vis, df_dx_net, 'EdgeColor', 'none');
title('Нейросетевая ∂f/∂x');
xlabel('x'); ylabel('y'); zlabel('∂f/∂x');
colorbar;

% ∂f/∂y аналитический
subplot(2,2,3);
surf(x_vis, y_vis, df_dy_vis, 'EdgeColor', 'none');
title('Аналитическая ∂f/∂y');
xlabel('x'); ylabel('y'); zlabel('∂f/∂y');
colorbar;

% ∂f/∂y нейросеть
subplot(2,2,4);
surf(x_vis, y_vis, df_dy_net, 'EdgeColor', 'none');
title('Нейросетевая ∂f/∂y');
xlabel('x'); ylabel('y'); zlabel('∂f/∂y');
colorbar;
