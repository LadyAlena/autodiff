function [f, grads] = func(x, y, W1, b1, W2, b2, W3, b3, W4, b4, in_min, in_max, out_min, out_max)
    % Нормализация входов
    x_norm = 2*(x - in_min(1))/(in_max(1) - in_min(1)) - 1;
    y_norm = 2*(y - in_min(2))/(in_max(2) - in_min(2)) - 1;
    
    % Объединяем входы
    input = [x_norm; y_norm];

    % Прямое распространение
    L1 = tanh(W1 * input + b1);
    L2 = tanh(W2 * L1 + b2);
    L3 = tanh(W3 * L2 + b3);
    z_norm = W4 * L3 + b4;
    
    % Денормализация выхода
    f = (z_norm + 1)*(out_max - out_min)/2 + out_min;
    
    % Вычисление градиентов
    [grad_x, grad_y] = dlgradient(f, x, y);
    grads = {grad_x, grad_y};
end