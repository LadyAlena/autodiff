function [f, grads] = func(x, y)
    f = x^2 + y^2 + x*y;
    [grad_x, grad_y] = dlgradient(f, x, y); % Частные производные
    grads = {grad_x, grad_y};
end
