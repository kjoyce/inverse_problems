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
%figure(1) 
%  imagesc(x_true), colorbar, colormap(1-gray)
%figure(2)
%  imagesc(b), colorbar, colormap(1-gray)
%figure(3)
%  imagesc((A1\b)/A2), colorbar, colormap(1-gray)

% SVD analysis
[U1,S1,V1] = svd(A1);
[U2,S2,V2] = svd(A2);
dS1 = diag(S1);
dS2 = diag(S2);
dS1dS2 = dS2*dS1';
Utb = (U1'*b)*U2;

% Find the UPRE choice for alpha (see Section 2.2) and plot the
% reconstruction
alpha_flag =1;% input(' Enter 1 for UPRE and 2 for DP regularization parameter selection. ');
if alpha_flag == 1
    RegParam_fn = @(a) a^2*sum(sum((Utb.^2)./(dS1dS2.^2+a).^2))+2*sigma^2*sum(sum(dS1dS2.^2./(dS1dS2.^2+a)));
elseif alpha_flag == 2
    RegParam_fn = @(a) (a^2*sum(sum((Utb.^2)./(dS1dS2.^2+a).^2))-n*sigma^2)^2;
end
gcv_fn = @(a) a^2*sum(Utb(:).^2./(dS1dS2(:)+a).^2)/(n-sum(dS1dS2(:)./(dS1dS2(:)+a)));
gcv_alpha = fminbnd(gcv_fn,0,1);

alpha = fminbnd(RegParam_fn,0,1);
xalpha = V1*((dS1dS2./(dS1dS2.^2+alpha)).*Utb)*V2';

gcv_sol = V1*((dS1dS2./(dS1dS2.^2+gcv_alpha)).*Utb)*V2';


%figure(4)
%  imagesc(xalpha), colorbar, colormap(1-gray)

aa = logspace(-7,0);
figure(1)
  subplot(2,2,1), imagesc(x_true), colorbar, colormap(1-gray), title('True image')
  subplot(2,2,2), imagesc(b), colorbar, colormap(1-gray),title('Blurred Noisey Image')
  subplot(2,2,3), imagesc(gcv_sol), colorbar, colormap(1-gray), title(sprintf('GCV reconstruction,  \\alpha = %.4e',gcv_alpha))
  subplot(2,2,4), loglog(aa,arrayfun(gcv_fn,aa),'b-'), xlim([aa(1),aa(end)]), title('GCV curve')

%set(gcf, 'PaperPosition', [0 0 5 5]); %Position plot at left hand corner with width 5 and height 5.
%set(gcf, 'PaperSize', []); %Set the paper to have width 5 and height 5.
saveas(gcf, 'images', 'pdf') %Save figure
