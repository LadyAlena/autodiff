clear, clc;
x = dlarray(1);
y = dlarray(2);
[f, grads] = dlfeval(@func, x, y);

% Извлекаем данные из dlarray
f_value = extractdata(f);
grad_x_value = extractdata(grads{1});
grad_y_value = extractdata(grads{2});

% Выводим результаты
disp(['f = ', num2str(f_value)]);
disp(['∂f/∂x = ', num2str(grad_x_value)]);
disp(['∂f/∂y = ', num2str(grad_y_value)]);