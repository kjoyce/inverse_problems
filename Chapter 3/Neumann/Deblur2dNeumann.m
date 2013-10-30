%  
%  Image deblurring inverse problem with periodic boundary conditions.
%
%% Generate data and build regularization matrix.
  clear all, close all
  load GaussianBlur440_normal
  %load satellite.mat
  x_true = 100*f_true/max(f_true(:)); 
  clear PSF f_true beta g sigma 
  [nx,ny]=size(x_true);
  % Generate data on 256^2 grid w/ periodic BCs, then restrict to 128^2.
  [PSF,center] = psfGauss([nx,ny],[2,2]);
  Ax = real(ifft2(fft2(fftshift(PSF)).*fft2(x_true)));
  Ax=Ax(101:228,101:228);
  x_true = x_true(101:228,101:228);
  [nx,ny] = size(Ax);
  err_lev = 2;
  noise = err_lev/100 * norm(Ax(:)) / sqrt(nx*ny);%input(' The standard deviation of the noise = ');
  b = Ax + noise*randn(nx,ny);
  figure(1), imagesc(x_true), colormap(gray), colorbar 
  figure(2), imagesc(b,[0,max(x_true(:))]), colormap(gray), colorbar 
  
  % Use 128^2 PSF model with reflective BCs
  PSF = PSF(65:192,65:192); center=[65,65];
  e1 = zeros(size(PSF)); e1(1,1) = 1;
  khat = dct2( dctshift(PSF, center) ) ./ dct2(e1);
  khatCirc = fft2(fftshift(PSF));

  % Construct Fourier representer for discrete Laplacian L.
  ls = zeros(size(b));
  ls(center(1)-1:center(1)+1,center(2)-1:center(2)+1)=[0 -1 0;-1 4 -1;0 -1 0];
  lhat = dct2( dctshift(ls, center) )./ dct2(e1);
  lhatCirc = fft2(fftshift(ls));
  
  % Compute the Tikhonov regularized solution with Neumann BC
  alpha  =input(' alpha = ');
  xalpha = idct2(khat.*dct2(b)./(khat.^2 + alpha*lhat));
  figure(3), imagesc(xalpha,[0,max(x_true(:))]), colormap(gray), colorbar

  % Compute the Tikhonov regularized solution with Circulant BC
  xalpha = real(ifft2(khatCirc.*fft2(b)./(abs(khatCirc).^2 + alpha*lhat)));
  figure(4), imagesc(xalpha,[0,max(x_true(:))]), colormap(gray), colorbar
