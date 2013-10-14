%  
%  1d image deblurring inverse problem with Dirichlet boundary conditions.
%
clear all, close all
n = 80; %%%input(' No. of grid points = ');
h = 1/n;
t = [0:h:1-h]';
sig = .05; %%%input(' Kernel width sigma = ');
kernel = (-100*t+10).*(t<=1/10);
A = toeplitz(kernel)*h;

% Set up true solution x_true and data b = A*x_true + error.
x_true = .75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4;
x_true = x_true/norm(x_true);
Ax = A*x_true;
err_lev = 2; %%%input(' Percent error in data = ');
sigma = err_lev/100 * norm(Ax) / sqrt(n);
eta =  sigma * randn(n,1);
b = Ax + eta;
figure(1), 
  plot(t,x_true,'k',t,b,'ko')
  %legend('true image','blurred, noisy data','Location','NorthWest')

[U,S,V] = svd(A);
dS = diag(S); dS2 = dS.^2; 
Utb = U'*b;

GCV_fn = @(a) sum((a^2*Utb.^2)./(dS2+a).^2)/(n-sum(dS2./(dS2+a)))^2;
Lcurve_fn = @(alpha) - curvatureLcurve(alpha,A,U,S,V,b);

%figure(2);
%aph = logspace(-13,-1);
%loglog(aph, arrayfun(GCV_fn,aph));
alpha_gcv = fminbnd( GCV_fn , 1e-6, 1e-1)
alpha_lcv = fminbnd( Lcurve_fn , 1e-6, 1e-1)


figure(2);
plot(t,x_true,'k',...
     t,(A'*A + alpha_gcv*eye(n))\A'*b,'b:',...
     t,(A'*A + alpha_lcv*eye(n))\A'*b,'g');
title('Tikhonov filter')
legend('true','GCV','L curve')


% Find the UPRE choice for k (see Section 2.2)
U_fn = @(k) norm(Utb(k+1:n))^2+2*sigma^2*k;
Uvals = zeros(n,1);
for i=1:n, Uvals(i)=U_fn(i); end
upre_k = find(Uvals == min(Uvals))

% Find the DP choice for k (see Section 2.2)
D_fn = @(k) (norm(Utb(k+1:n))^2-n*sigma^2)^2;
Dvals = zeros(n,1);
for i=1:n, Dvals(i)=D_fn(i); end
dp_k = find(Dvals == min(Dvals))

% Now compute the regularized solution for TSVD
phi = zeros(n,1); phi(1:upre_k)=1; 
idx = (dS>0);
dSfilt = zeros(size(dS));
dSfilt(idx) = phi(idx)./dS(idx); 
upre_xfilt = V*(dSfilt.*(U'*b));

phi = zeros(n,1); phi(1:dp_k)=1; 
idx = (dS>0);
dSfilt = zeros(size(dS));
dSfilt(idx) = phi(idx)./dS(idx); 
dp_xfilt = V*(dSfilt.*(U'*b));

figure(3);
plot(t,x_true,'k',...
     t,upre_xfilt,'b',...
     t,dp_xfilt,'g');
title('TSVD filter')
legend('true','upre','dp')
