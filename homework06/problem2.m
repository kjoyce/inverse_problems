clear all, close all;
% b
n = 1000
v = randn(2,n);
mu = ones(2,n);
C = [2 -1;-1 2];
R = chol(C);
w = mu + R'*v;
plot(w(1,:),w(2,:),'.') ,xlim([-5,7]) ,ylim([-5,7]);
hold on;

% c
vv = R\(w - mu);
xi = sum(vv.^2);
alpha = .05;
l1 = chi2inv((1-alpha),2);
sum(xi <= l1)/length(xi)
alpha = .01;
l2 = chi2inv((1-alpha),2);
sum(xi <= l2)/length(xi)


[x y] = meshgrid(-5:.1:7,-5:.1:7);
xxyy = R\(-ones(2,size(x(:),1)) + [x(:) y(:)]');
xx = reshape(xxyy(1,:),size(x,1),size(x,1)); 
yy = reshape(xxyy(2,:),size(x,1),size(x,1)); 
contour(x,y,chi2pdf(xx.^2 + yy.^2,2),[chi2pdf(l1,2) chi2pdf(l2,2)])
