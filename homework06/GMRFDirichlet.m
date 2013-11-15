% Simulation from a 1D and 2D GMRF

%% 1D case
%  Build the A matrix corresponding to forward or backward difference
%     -x(i-1)-x(i+1)+2*x(i)
%      x(0) = x(N+1) = 0
n=128;
one_vec = ones(n,1);

L1D = spdiags([-one_vec 2*one_vec -one_vec],[-1 0 1],n,n);
L1D(1,n) = -1;
L1D(n,1) = -1;
[V,D] = eig(full(L1D));
R = V*sqrt(D)\V;

v = randn(n,5);
% Plot iid increment samples
samps = R\v;
figure(1)
  plot(samps,'k')
  axis([0 128 min(samps(:)) max(samps(:))])

%% 2D case
%  Build the A matrix corresponding to the neighborhood relationship
%     -x(i-1,j)-x(i+1,j)-x(i,j-1)-x(i,j+1)+4*x(i) 
%      x(0,j) = x(N+1,j) = x(i,0) = x(i,N+1) = 0
nx = 128;
ny = 128;
lh=zeros(nx,ny); lh(1,1)=-1; lh(1,2)=1; lhhat=fft2(lh);
lv=zeros(nx,ny); lv(1,1)=-1; lv(nx,1)=1; lvhat=fft2(lv);
lhat = abs(lhhat).^2+abs(lvhat).^2;
ii = find(lhat>0);
lhat_inv = zeros(size(lhat));
lhat_inv(ii) = 1./lhat(ii);
 
samp = ifft2(sqrt(lhat_inv).*(fft2(randn(nx,ny))));
figure(2)
  imagesc(reshape(samp,nx,ny)), colorbar, colormap(gray)

  
