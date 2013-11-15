%  
%  
%  Tomography Inverse Problem
%
%% Generate data from Shepp-Logan phantom
  clear all, close all
  n         = 100;
  x_true    = phantom('Modified Shepp-Logan',n);
  ntheta    = 101;
  theta     = linspace(-pi/2,pi/2,ntheta);
  nz        = 99;
  z         = linspace(-0.49,0.49,nz);
  [Z,Theta] = meshgrid(z,theta);
  A         = Xraymat(Z(:),Theta(:),n);
  Ax        = A*x_true(:);
  err_lev   = 2;
  noise     = err_lev/100 * norm(Ax(:)) / sqrt(ntheta*nz);
  b         = reshape(Ax,ntheta,nz) + noise*randn(ntheta,nz);
  % Data display
  figure(1), imagesc(x_true), colormap(1-gray), colorbar
  figure(2), imagesc(b), colormap(1-gray), colorbar


%x = ones(size(A(1,:)))';  % initial guess
x = A(1,:)';
figure()
j = 1;
i = 1;
h = waitbar(0,'Percentage to DP')
while 1
  i = i + 1;
  x  = x + (b(i) - A(i,:)*x)/norm(A(i,:))^2 * A(i,:)';
  dp = norm( A*x - b(:) )^2/(size(A,1)*noise^2);
  if mod(i,100) == 0
    imagesc(reshape(x,size(x_true))), colormap(1-gray),colorbar;
    title(sprintf('i = %d, dp = %.3f',i,dp)), drawnow;
    waitbar(1/dp)
  end
  if i == size(A,1)
    i = 0;
    j = j+1;
  end
  if dp <= 1
    break
  end
end 
close(h)
