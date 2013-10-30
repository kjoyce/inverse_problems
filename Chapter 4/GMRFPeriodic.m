% Simulation from a 1D and 2D GMRF
clear all, close all
%% 1D case
n=128;
one_vec = ones(n,1);
D = spdiags([-one_vec one_vec],[0 1],n,n); D(n,1) = 1;
L1D = D'*D+sqrt(eps)*speye(n,n);
v = D'*randn(n,5);
% Plot iid increment samples
samps = L1D\v;
samps = samps-repmat(mean(samps,1),n,1);
figure(1)
  plot(samps,'k')
  axis([0 128 min(samps(:)) max(samps(:))])

%% 2D case
% Construct Fourier representer for 
% discrete Laplacian L.
%l = zeros(n,n); 
%l(1, 1) =  4; l(2 ,1) = -1; 
%l(n,1) = -1; l(1 ,2) = -1; 
%l(1,n) = -1; lhat = real(fft2(l));
% discrete horizontal and vertical forward difference derivatives.
lh=zeros(n,n); lh(1,1)=-1; lh(1,2)=1; lhhat=fft2(lh);
lv=zeros(n,n); lv(1,1)=-1; lv(n,1)=1; lvhat=fft2(lv);
lhat = abs(lhhat).^2+abs(lvhat).^2;

v = real(ifft2(sqrt(lhat).*fft2(randn(n,n))));
samp = real(ifft2(fft2(v)./(lhat+sqrt(eps))));
figure(2)
  imagesc(samp), colorbar, colormap(gray)

  