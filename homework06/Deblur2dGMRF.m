%  
%  2d image deblurring inverse problem with periodic boundary conditions.
%%
clear all, close all
load satellite
x_true = x_true/max(x_true(:));
[nx,ny] = size(x_true);
N = nx*ny;
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
sigma = err_lev/100 * norm(Ax(:)) / sqrt(N);
rng(0)
eta =  sigma * randn(nx,ny);
b = Ax + eta;
bhat = fft2(b);
figure(1) 
  imagesc(x_true), colorbar, colormap(1-gray)
figure(2)
  imagesc(b), colorbar, colormap(1-gray)
% Construct Fourier representer for discrete Laplacian L.
%%
lh=zeros(nx,ny); lh(1,1)=-1; lh(1,2)=1; lhhat=fft2(lh);
lv=zeros(nx,ny); lv(1,1)=-1; lv(nx,1)=1; lvhat=fft2(lv);
lhat = abs(lhhat).^2+abs(lvhat).^2;
  
% Find the discrepancy choice for alpha (see Section 2.2) and plot the
% reconstruction
G_fn=@(alpha)(sum(sum((alpha^2*abs(lhat).^2.*abs(bhat).^2)./(abs(ahat).^2+alpha*lhat).^2))) ...
             / (N-sum(sum(abs(ahat).^2./(abs(ahat).^2+alpha*lhat))))^2;
alpha =  fminbnd(G_fn,0,1);
xalpha = real(ifft2((conj(ahat)./(abs(ahat).^2+alpha*lhat).*bhat)));
figure(4)
  imagesc(xalpha), colorbar, colormap(1-gray)
  
