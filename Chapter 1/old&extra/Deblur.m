%  
%  1d image deblurring inverse problem with Dirichlet boundary conditions.
%
clear all, close all
load satellite
[nx,ny] = size(x_true);
n = nx*ny;
h = 1/nx;
t = [h/2:h:1-h/2]';
sig = .05; %%%input(' Kernel width sigma = ');
kernel1 = (1/sqrt(pi)/sig) * exp(-(t-h/2).^2/sig^2);
kernel2 = kernel1;
A1 = toeplitz(kernel1)*h;
A2 = toeplitz(kernel2)*h;

Ax = (A1*x_true)*A2';
err_lev = 2; %%%input(' Percent error in data = ');
sigma = err_lev/100 * norm(Ax(:)) / sqrt(n);
%randn('state',1)
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
S1 = sparse(S1); S2 = sparse(S2);
S = kron(S1,S2);
figure(4),
  semilogy(sort(diag(S),'descend'),'.k')
