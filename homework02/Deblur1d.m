%  
%  1d image deblurring inverse problem with Dirichlet boundary conditions.
%
clear all, close all
n = 80; %%%input(' No. of grid points = ');
h = 1/n;
t = [h/2:h:1-h/2]';
sig = .05; %%%input(' Kernel width sigma = ');
kernel = (1/sqrt(pi)/sig) * exp(-(t-h/2).^2/sig^2);
A = toeplitz(kernel)*h;

% Set up true solution x_true and data b = A*x_true + error.
x_true = 50*(.75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4);
x_true = x_true/norm(x_true);
Ax = A*x_true;
err_lev = 2; %%%input(' Percent error in data = ');
sigma = err_lev/100 * norm(Ax) / sqrt(n);
rng(0) % Use randn('seed',0) for old version of MATLAB
eta =  sigma * randn(n,1);
b = Ax + eta;
figure(1), 
  plot(t,x_true,'k',t,b,'ko')
  %legend('true image','blurred, noisy data','Location','NorthWest')
figure(2),
  plot(t,A\b,'k')

delta_b = norm(b-Ax)
delta_x = norm(x_true-A\b)

% SVD analysis
[U,S,V] = svd(A);
figure(3),
  semilogy(diag(S),'ko')

% Singular vector analysis
figure(4)
title('Right Singular Vectors')
for i=1:8
    subplot(2,4,i)
    plot(t,V(:,i))
    title(sprintf('v_{%d}',i))
end
figure(5)
title('Left Singular Vectors')
for i=1:10:80
    subplot(2,4,(i-1)/10+1),
    plot(t,U(:,i)),
    title(sprintf('u_{%d}',i));
end
figure(6),
semilogy(1:n,diag(S),'ko', 1:n,abs(U'*b),'b*', 1:n,abs(U'*b)./diag(S), 'r.')
title('Noisy Data')
legend('Singular Values', 'u_i^Tb', 'ratio','Location','SouthWest')

figure(7)
semilogy(1:n,diag(S),'ko', 1:n,abs(U'*Ax),'b*', 1:n,abs(U'*Ax)./diag(S), 'r.','Linewidth',2)
title('Data with No Noise')
legend('Singular Values', 'u_i^Tb', 'ratio','Location','SouthWest')

figure(8)
semilogy(sigma^2./diag(S).^2,'k.')
