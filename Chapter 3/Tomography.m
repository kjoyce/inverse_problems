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