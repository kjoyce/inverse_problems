%  
%  1d image deblurring inverse problem with Dirichlet boundary conditions.
%%
clear all, close all
n      = 120; %%%input(' No. of grid points = ');
h      = 1/n;
t      = [h/2:h:1-h/2]';
sig    = .02; %%%input(' Kernel width sigma = ');
kernel = (1/sqrt(pi)/sig) * exp(-(t-h/2).^2/sig^2);
A      = toeplitz(kernel)*h;
% Restrict the domain
ii     = find(t>.15 & t<.85);
A      = A(ii,:);

% Set up true solution x_true and data b = A*x_true + error.
x_true = .75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4;
x_true = x_true/norm(x_true);
Ax = A*x_true;
err_lev = 2; %%%input(' Percent error in data = ');
sigma = err_lev/100 * norm(Ax) / sqrt(length(Ax));
eta =  sigma * randn(length(Ax),1);
b = Ax + eta;

%% Compute Tikhonov solution for the data driven boundary conditions
[U,S,V] = svd(A,'econ');
dS = diag(S); dS2 = dS.^2; 
Utb = U'*b;
% Take DP choice of regularization parameter.
DP_fn = @(a) (sum((a^2*Utb.^2)./(dS2+a).^2)-n*sigma^2)^2;
GCV_fn = @(a) sum((a^2*Utb.^2)./(dS2+a).^2)/(n-sum(dS2./(dS2+a)))^2;
UPRE_fn = @(a) sum((a^2*Utb.^2)./(dS2+a).^2)+2*sigma^2*sum(dS2./(dS2+a));
alpha_dp = fminbnd(DP_fn,0,1);
alpha_gcv = fminbnd(GCV_fn,0,1);
alpha_upre = fminbnd(UPRE_fn,0,1);
% Now compute the regularized solution for TSVD
dSfilt_gcv  = dS./(dS.^2+alpha_gcv );
dSfilt_upre = dS./(dS.^2+alpha_upre);
dSfilt_dp = dS./(dS.^2+alpha_dp);
xfilt_gcv  = V*(dSfilt_gcv .*(U'*b));
xfilt_upre = V*(dSfilt_upre.*(U'*b));
xfilt_dp = V*(dSfilt_dp.*(U'*b));

% Compute Tikhonov solution again for the zero BCs
Azero = A(:,ii);
[U,S,V] = svd(Azero);
dS = diag(S); dS2 = dS.^2; 
Utb = U'*b;
% Take DP choice of regularization parameter.
DP_fn = @(a) (sum((a^2*Utb.^2)./(dS2+a).^2)-n*sigma^2)^2;
GCV_fn = @(a) sum((a^2*Utb.^2)./(dS2+a).^2)/(n-sum(dS2./(dS2+a)))^2;
UPRE_fn = @(a) sum((a^2*Utb.^2)./(dS2+a).^2)+2*sigma^2*sum(dS2./(dS2+a));
  
zbc_alpha_dp = fminbnd(DP_fn,0,1);
zbc_alpha_gcv = fminbnd(GCV_fn,0,1);
zbc_alpha_upre = fminbnd(UPRE_fn,0,1);
% Now compute the regularized solution for TSVD
dSfilt_dp = dS./(dS.^2+zbc_alpha_dp);
dSfilt_gcv = dS./(dS.^2+zbc_alpha_gcv);
dSfilt_upre = dS./(dS.^2+zbc_alpha_upre);
xfiltZeroBC_dp = V*(dSfilt_dp.*(U'*b));
xfiltZeroBC_gcv = V*(dSfilt_gcv.*(U'*b));
xfiltZeroBC_upre = V*(dSfilt_upre.*(U'*b));
% Create plots
figure(1), 
  plot(t,x_true,'k',t(ii),b,'ko'), axis([t(1),t(end), -0.05, 0.35])
  legend('true image','blurred, noisy data','Location','NorthWest')
figure(2)
  plot(t,x_true,'k-',t(ii),xfiltZeroBC_gcv,'bd-',t(ii),xfiltZeroBC_upre,'k+-',t(ii),xfiltZeroBC_dp,'g.-')
  axis([t(1),t(end), -0.05, 0.35])
  legend('true image',sprintf('GCV alpha = %.3e',zbc_alpha_gcv),sprintf('UPRE alpha = %.3e',zbc_alpha_upre),sprintf('DP alpha = %.3e',zbc_alpha_dp),'Location','NorthWest')
  title('Zero Boundary Conditions')

figure(3)
  plot(t,x_true,'k-',t(ii),xfilt_gcv(ii),'bd-',t(ii),xfilt_upre(ii),'k+-',t(ii),xfilt_dp(ii),'g.-')
  axis([t(1),t(end), -0.05, 0.35])
  legend('true image',sprintf('GCV alpha = %.3e',alpha_gcv),sprintf('UPRE alpha = %.3e',alpha_upre),sprintf('dp alpha = %.3e',alpha_dp),'Location','NorthWest')
  title('Data Driven Boundary Conditions')
