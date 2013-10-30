% Simulation from a 1D and 2D GMRF

%% 1D case
%  Build the A matrix corresponding to forward or backward difference
%     -x(i-1)-x(i+1)+2*x(i)
%      x(0) = x(N+1) = 0
n=128;
one_vec = ones(n,1);
L1D = spdiags([-one_vec 2*one_vec -one_vec],[-1 0 1],n,n);
R = chol(L1D);
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
N = nx*ny;
%  Set up matrix A. Use MATLAB's sparse storage.
Adiag = 4*ones(N,1);
Asubs = -ones(N,1);
Asubs1 = -ones(N,1);
Asuper1 = -ones(N,1);
% we must account for the zeroes in the sub and super diagonal
for i=1:ny, Asub1(i*nx) = 0; end
for i=0:ny-1, Asuper1(i*nx+1)=0; end
L2D = spdiags([Asubs,Asubs1,Adiag,Asuper1,Asubs],[-nx -1 0 1 nx],N,N);
 
v = randn(N,1);
R = chol(L2D);
samp = R\v;
figure(2)
  imagesc(reshape(samp,nx,ny)), colorbar, colormap(gray)

  