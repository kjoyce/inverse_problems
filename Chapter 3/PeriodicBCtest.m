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
  h = 1/nx;
  x = [-0.5+h/2:h:0.5-h/2]';
  [X,Y]=meshgrid(x);
  sig = 2*h;
  kernel = exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
  kernel = kernel/sum(sum(kernel));
  Ax = real(ifft2(fft2(fftshift(kernel)).*fft2(x_true)));
  Ax=Ax(101:228,101:228);
  x_true = x_true(101:228,101:228);
  [nx,ny] = size(Ax);
  err_lev = 2;
  noise = err_lev/100 * norm(Ax(:)) / sqrt(nx*ny);%input(' The standard deviation of the noise = ');
  b = Ax + noise*randn(nx,ny);
  figure(1), imagesc(x_true), colormap(gray), colorbar 
  figure(2), imagesc(b,[0,max(x_true(:))]), colormap(gray), colorbar 
  
  % Use 128^2 PSF model with periodic BCs and GCV choice of alpha
  kernel = kernel(65:192,65:192); 

n = nx*ny
ahat = fft2(fftshift(kernel));
Amult = @(x) real(ifft2(ahat.*fft2(x)));
Ax = Amult(x_true);
bhat = fft2(b);
G_fn=@(a)(sum(sum((a^2*abs(bhat).^2)./(abs(ahat).^2+a).^2))) / ...
	 (n-sum(sum(abs(ahat).^2./(abs(ahat).^2+a))))^2;
gcv_alpha =  fminbnd(G_fn,0,1);

