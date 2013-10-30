% Simulation from a 1D and 2D GMRF

%% 1D case
% x_{i+1}-x_i ~ N(0,1)
n=128;
one_vec = ones(n,1);
D = spdiags([-one_vec one_vec],[0 1],n-1,n);
v = randn(n-1,5);
% Plot iid increment samples
samps = D\v;
figure(1)
  plot(samps)
% Plot independent increments
G=speye(n-1,n-1);
G(n/2,n/2) = 0.05;
samps = D\(G\v);
figure(2)
  plot(samps,'k')
  axis([0 128 min(samps(:)) max(samps(:))])
%% 2D iid increment
% x_{i+1,j}-x_{i,j} ~ N(0,1) and x_{i,j+1}-x_{i,j} ~ N(0,1)
I  = speye(n,n);
Dh = kron(I,D);
Dv = kron(D,I);
% Plot iid increment sample
L2D = Dh'*Dh + Dv'*Dv;
R = chol(L2D);
figure(3)
  imagesc(reshape(R\randn(n^2,1),n,n)), colormap(gray), colorbar
%% 2D independent increment sample
% First create an edge set using a circle. Then decrease the precision
% (increase the variance) at the edge for the increments.
x = [1/(n+1):1/(n+1):1-1/(n+1)];
[X,Y]=meshgrid(x);
Z = (X-.5).^2+(Y-.5).^2;
circle = (Z(:)<.1); % indicator function on a circle
ngradderiv = sqrt((Dh*circle).^2 + (Dv*circle).^2);
Gdiag = ones(n*(n-1),1);
Gdiag(ngradderiv > 0) = 0.001;
G = spdiags(Gdiag,0,n*(n-1),n*(n-1));
L2D = Dh'*(G*Dh) + Dv'*(G*Dv);
R = chol(L2D+sqrt(eps)*speye(n^2,n^2));
figure(4)
  imagesc(reshape(R\randn(n^2,1),n,n)), colormap(gray), colorbar
