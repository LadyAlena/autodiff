function f = interp2fast(xt, yt, ft, x, y)
% ----------------------------------------------------------------
%               ���������� ������������
% ----------------------------------------------------------------

nx = length(xt);
ny = length(yt);

% ����� ��������� �� X
for i = 1:nx-1
    if(x<xt(i+1)), break; end
end
% ����� ��������� �� Y
for j = 1:ny-1
    if(y<yt(j+1)), break; end
end

% ������������ X, ���� ��� �������
if(x<xt(1)), x = xt(1); end
if(x>xt(nx)), x = xt(nx); end
% ������������ Y, ���� ��� �������
if(y<yt(1)), y = yt(1); end
if(y>yt(ny)), y = yt(ny); end

f = ( ft(i,j)*(xt(i+1)-x)*(yt(j+1)-y) + ft(i,j+1)*(xt(i+1)-x)*(y-yt(j)) + ...
      ft(i+1,j)*(x-xt(i))*(yt(j+1)-y) + ft(i+1,j+1)*(x-xt(i))*(y-yt(j)) ) ...
      /(( xt(i+1)-xt(i) )*( yt(j+1)-yt(j) ));
