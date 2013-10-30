%  
%  1d image deblurring inverse problem with Dirichlet boundary conditions.
%
clear all, close all
rng(0) % For IPSE figures
n = 80; %%%input(' No. of grid points = ');
h = 1/n;
t = [h/2:h:1-h/2]';
sig = .05; %%%input(' Kernel width sigma = ');
kernel = (1/sqrt(pi)/sig) * exp(-(t-h/2).^2/sig^2);
A = toeplitz(kernel)*h;

% Set up true solution x_true and data b = A*x_true + error.
x_true = 50*(.75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4);
Ax = A*x_true;
err_lev = 2; %%%input(' Percent error in data = ');
sigma = err_lev/100 * norm(Ax) / sqrt(n);
eta =  sigma * randn(n,1);
b = Ax + eta;
figure(1), 
plot(t,x_true,'k',t,b,'ko','LineWidth',1)
legend('true image','blurred, noisy data')

% second derivative precision matrix for prior
  L = spdiags([-ones(n,1) 2*ones(n,1) -ones(n,1)],[-1 0 1],n,n);
% MCMC sampling
  nsamps = 1000;
  nruns = 5;
  xsamp = zeros(n,nsamps,nruns); 
  delsamp = zeros(nsamps,nruns); delsamp(1,:) = .5*rand(1,nruns);
  lamsamp = zeros(nsamps,nruns); lamsamp(1,:) = 2+6*rand(1,nruns);
  tic
  for i = 1:nsamps
    for k = 1:nruns
      h = waitbar(((i-1)*nruns+k)/(nsamps*nruns));
      %------------------------------------------------------------------
      % 1. Using conjugacy relationships, first sample the image. 
      R = chol(A'*A*lamsamp(i,k) + delsamp(i,k)*L);
      xsamp(:,i,k) = R \ (R'\(A'*b*lamsamp(i,k)) + randn(n,1));
      %------------------------------------------------------------------
      % 2. Using conjugacy, sample the noise precision lam=1/sigma^2, 
      % conjugate prior: lam~Gamma(a,1/t0)
      a0=1; t0=0.0001; % uninformative prior values
      lamsamp(i+1,k) = gamrnd(a0+n/2,1/(t0+norm(A*xsamp(:,i,k)-b)^2/2));
      %------------------------------------------------------------------
      % 3. Using conjugacy, sample regularization precisions delta, 
      % conjugate prior: delta~Gamma(a1,1/t1);
      a1=1; t1=0.0001; % uninformative prior values
      delsamp(i+1,k) = gamrnd(a1+n/2,1./(t1+xsamp(:,i,k)'*(L*xsamp(:,i,k))/2));
    end
    %------------------------------------------------------------------
    % 4. Test mixing and convergence of the chains every nn samples.
    nn = 100;
    if mod(i,nn)==0 & nruns>1
      jj=floor(i/2)+1:i; ns = length(jj);
      xx = zeros(n+2,ns,nruns);
      xx(1:n,:,:)=xsamp(:,jj,:);
      xx(n+1,:,:)=lamsamp(jj,:);
      xx(n+2,:,:)=delsamp(jj,:);
      mean_j = sum(xx,2)/ns; % integrate each chain
      mean_ij = sum(mean_j,3)/nruns; % integrate between chains
      B = ns/(nruns-1)*sum((mean_j-repmat(mean_ij,[1,1,nruns])).^2,3);
      var_j = (1/(ns-1))*sum((xx-repmat(mean_j,[1,ns,1])).^2,2);
      W = sum(var_j,3)/nruns;
      Rhat = sqrt( ((ns-1)*W/ns+B/ns) ./ W );
      fprintf('Sample %d out of %d total; max Rhat = %2.3f.\n',i,nsamps,max(Rhat))
    end
  end
  toc
  close(h)
  % Use last half of samples for analysis
  xsamp = xsamp(:,floor(nsamps/2):nsamps-1,:); 
  delsamp=delsamp(floor(nsamps/2):nsamps,:); 
  lamsamp=lamsamp(floor(nsamps/2):nsamps,:);
  qlam = quantile(lamsamp(:),[0.025,.975]);
  fprintf('1/sigma^2 = %2.3f; 95 percent credibility interval: [%2.3f, %2.3f]\n',1/sigma^2,qlam(1),qlam(2))
  figure(2), 
    q = quantile(xsamp(:,:)',[0.025,0.975]);
    x_mean = mean(xsamp(:,:)')';
    plot(t,x_mean,'k',t,x_true,'-.k',t,q(2,:),'--k',t,q(1,:),'--k')
    legend('MCMC sample mean','true image','95% credibility bounds','Location','North')
  relative_error = norm(x_true-x_mean)/norm(x_true)
  figure(3), colormap(1-gray)
    subplot(3,1,1), hist(delsamp(:),25), title('\delta, the prior precision')
    subplot(3,1,2), hist(lamsamp(:),25), title('\lambda, the noise precision')
    subplot(3,1,3), hist(delsamp(:)./lamsamp(:),25), title('\alpha=\delta/\lambda, the regularization parameter')