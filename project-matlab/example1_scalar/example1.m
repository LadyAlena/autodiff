clear, clc;
x = dlarray(1);
[f, grad] = dlfeval(@func, x); % Запуск вычислений

% Извлечение данных из dlarray и вывод результатов
f_value = extractdata(f);
grad_value = extractdata(grad);

disp(['f = ', num2str(f_value)]);
disp(['∂f/∂x = ', num2str(grad_value)]);
