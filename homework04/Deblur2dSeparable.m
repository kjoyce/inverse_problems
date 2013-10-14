%  
%  2d image deblurring inverse problem with separable kernel.
%
clear all, close all
load satellite
[nx,ny] = size(x_true);
x_true = x_true/max(x_true(:));
n = nx*ny;
h = 1/nx;
t = [h/2:h:1-h/2]';
sig = .02; %%%input(' Kernel width sigma = ');
kernel1 = (1/2/sqrt(pi)/sig) * exp(-(t-h/2).^2/2/sig^2);
kernel2 = kernel1;
A1 = toeplitz(kernel1)*h;
A2 = toeplitz(kernel2)*h;

Ax = (A1*x_true)*A2';
err_lev = 2; %%%input(' Percent error in data = ');
sigma = err_lev/100 * norm(Ax(:)) / sqrt(n);
rng(0)
eta =  sigma * randn(nx,ny);
b = Ax + eta;
figure(1) 
  imagesc(x_true), colorbar, colormap(1-gray)
figure(2)
  imagesc(b), colorbar, colormap(1-gray)
figure(3)
  imagesc((A1\b)/A2), colorbar, colormap(1-gray)

% SVD analysis
[U1,S1,V1] = svd(A1);
[U2,S2,V2] = svd(A2);
dS1 = diag(S1);
dS2 = diag(S2);
dS1dS2 = dS1*dS2';
Utb = (U1'*b)*U2;

% Find the UPRE choice for alpha (see Section 2.2) and plot the
% reconstruction
alpha_flag = input(' Enter 1 for UPRE and 2 for DP regularization parameter selection. ');
if alpha_flag == 1
    RegParam_fn = @(a) a^2*sum(sum((Utb.^2)./(dS1dS2.^2+a).^2))+2*sigma^2*sum(sum(dS1dS2.^2./(dS1dS2.^2+a)));
elseif alpha_flag == 2
    RegParam_fn = @(a) (a^2*sum(sum((Utb.^2)./(dS1dS2.^2+a).^2))-n*sigma^2)^2;
end
alpha = fminbnd(RegParam_fn,0,1);
xalpha = V1*((dS1dS2./(dS1dS2.^2+alpha)).*Utb)*V2';

figure(4)
  imagesc(xalpha), colorbar, colormap(1-gray)
