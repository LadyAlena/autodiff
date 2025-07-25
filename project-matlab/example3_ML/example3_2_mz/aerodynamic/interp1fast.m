function f = interp1fast(xt, ft, x)
% ----------------------------------------------------------------
%               �������� ������������
% ----------------------------------------------------------------

nx = length(xt);

% ����� ���������
for i = 1:nx-1
    if(x<xt(i+1)), break; end
end

% ������������, ���� ��� �������
if(x<xt(1)), x = xt(1); end
if(x>xt(nx)), x = xt(nx); end

f = ( ft(i)*(xt(i+1)-x) + ft(i+1)*(x-xt(i)) )/(xt(i+1)-xt(i));