function [y, dy] = func(x)
    y = x.^2 + sin(x); % Функция
    dy = dlgradient(y, x); % Градиент (dy/dx)
end