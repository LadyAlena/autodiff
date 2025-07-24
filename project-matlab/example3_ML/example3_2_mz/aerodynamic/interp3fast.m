function f = interp3fast(xt, yt, zt, ft, x, y, z)
% ----------------------------------------------------------------
%               ����������� ������������
% ----------------------------------------------------------------

nx = length(xt);
ny = length(yt);
nz = length(zt);

% ����� ��������� �� X
for i = 1:nx-1
    if(x<xt(i+1)), break; end
end
% ����� ��������� �� Y
for j = 1:ny-1
    if(y<yt(j+1)), break; end
end
% ����� ��������� �� Z
for k = 1:nz-1
    if(z<zt(k+1)), break; end
end

% ������������ X, ���� ��� �������
if(x<xt(1)), x = xt(1); end
if(x>xt(nx)), x = xt(nx); end
% ������������ Y, ���� ��� �������
if(y<yt(1)), y = yt(1); end
if(y>yt(ny)), y = yt(ny); end
% ������������ Z, ���� ��� �������
if(z<zt(1)), z = zt(1); end
if(z>zt(nz)), z = zt(nz); end

f = ( ft(i,j,k)*(xt(i+1)-x)*(yt(j+1)-y)*(zt(k+1)-z) + ...  % �������� 1
      ft(i,j+1,k)*(xt(i+1)-x)*(y-yt(j))*(zt(k+1)-z) + ...
      ft(i+1,j,k)*(x-xt(i))*(yt(j+1)-y)*(zt(k+1)-z) + ...
      ft(i+1,j+1,k)*(x-xt(i))*(y-yt(j))*(zt(k+1)-z) + ...
      ft(i,j,k+1)*(xt(i+1)-x)*(yt(j+1)-y)*(z-zt(k)) + ...  % �������� 2
      ft(i,j+1,k+1)*(xt(i+1)-x)*(y-yt(j))*(z-zt(k)) + ...
      ft(i+1,j,k+1)*(x-xt(i))*(yt(j+1)-y)*(z-zt(k)) + ...
      ft(i+1,j+1,k+1)*(x-xt(i))*(y-yt(j))*(z-zt(k)) ) ...
      /(( xt(i+1)-xt(i) )*( yt(j+1)-yt(j) )*( zt(k+1)-zt(k) ));