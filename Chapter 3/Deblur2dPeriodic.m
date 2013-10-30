%  
%  2d image deblurring inverse problem with periodic boundary conditions.
%
clear all, close all
load satellite
x_true = x_true/max(x_true(:));
[nx,ny] = size(x_true);
n = nx*ny;
h = 1/nx;
x = [-0.5+h/2:h:0.5-h/2]';
[X,Y]=meshgrid(x);
sig = 0.02;
kernel = exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
kernel = kernel/sum(sum(kernel));
ahat = fft2(fftshift(kernel));

Amult = @(x) real(ifft2(ahat.*fft2(x)));
Ax = Amult(x_true);
err_lev = 2; %%%input(' Percent error in data = ')
sigma = err_lev/100 * norm(Ax(:)) / sqrt(n);
rng(0)
eta =  sigma * randn(nx,ny);
b = Ax + eta;
bhat = fft2(b)./n;
figure(1) 
  imagesc(x_true), colorbar, colormap(1-gray)
figure(2)
  imagesc(b), colorbar, colormap(1-gray)
figure(3)
  imagesc(real(ifft2(bhat./ahat))), colorbar, colormap(1-gray)
  
% Find the discrepancy choice for alpha (see Section 2.2) and plot the
% reconstruction
%alpha_flag = 1;%input(' Enter 1 for GCV and 2 for L-curve regularization parameter selection. ');
G_fn=@(a)(sum(sum((a^2*abs(bhat).^2)./(abs(ahat).^2+a).^2))) / ...
	 (n-sum(sum(abs(ahat).^2./(abs(ahat).^2+a))))^2;
gcv_alpha =  fminbnd(G_fn,0,1);
C_fn = @(alpha) - curvatureLcurve(alpha,ahat,b);
lc_alpha = fminbnd(C_fn,0,1);

upre_fn = @(a) a^2*sum(sum((abs(bhat).^2)./(abs(ahat).^2+a).^2))+2*sigma^2*sum(sum(abs(ahat).^2./(abs(ahat).^2+a)));
upre_alpha = fminbnd(upre_fn,0,1)
dp_fn = @(a) (a^2*sum(sum((abs(bhat).^2)./(abs(ahat).^2+a).^2))-n*sigma^2)^2;
dp_alpha = fminbnd(dp_fn,0,1)

xalpha = @(a) real(ifft2((conj(ahat)./(abs(ahat).^2+a).*bhat)));

figure(4)
  subplot(2,2,1), imagesc(xalpha(gcv_alpha)), colorbar, colormap(1-gray)
  subplot(2,2,2), imagesc(xalpha(lc_alpha)), colorbar, colormap(1-gray)
  subplot(2,2,3), imagesc(xalpha(upre_alpha)), colorbar, colormap(1-gray)
  subplot(2,2,4), imagesc(xalpha(dp_alpha)), colorbar, colormap(1-gray)
  
