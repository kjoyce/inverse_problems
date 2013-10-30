%  
% PSF reconstruction problem.
%
clear all, close all
n       = 40; %%%input(' No. of grid points = ');
h       = 1/n;
t       = [-1+h:h:1-h]';
kernel  = zeros(size(t));
% Create the left-half of the PSF
sig1    = .02; %%%input(' Kernel width sigma = ');
kernel(1:n) = exp(-t(1:n).^2/sig1^2);
% Create the right-half of the PSF
sig2    = .08;
kernel(n:end) = exp(-t(n:end).^2/sig2^2);
% Create the normalized kernel and plot
kernel  = kernel/sum(kernel)/h;
figure(1), plot(t,kernel), title('kernel')
