%  
%  1d image deblurring inverse problem with Dirichlet boundary conditions.
%
clear all, close all
n = 80; %%%input(' No. of grid points = ');
h = 1/n;
t = [h/2:h:1-h/2]';
A = tril(ones(n,n))*h;
  
% Set up true solution x_true and data b = A*x_true + error.
x_true = .75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4;
x_true = x_true/norm(x_true);
Ax = A*x_true;
err_lev = 2; %%%input(' Percent error in data = ');
sigma = err_lev/100 * norm(Ax) / sqrt(n);
%randn('state',1)
eta =  sigma * randn(n,1);
b = Ax + eta;
figure(1), 
  plot(t,x_true,'k',t,b,'ko')
  %legend('true image','blurred, noisy data','Location','NorthWest')
  
% Compute Landweber solution
tau = 0.6;
x = zeros(n,1);
err = norm(x_true-x);
iterflag = 0;
i = 0;
while iterflag == 0
    i = i+1;
    x = x - tau*A'*(A*x-b);
    errnew = norm(x_true-x);
    if errnew >= err
        iterflag = 1;
    else
        err = errnew;
    end
end
figure(2), 
  plot(t,x_true,'b',t,x,'k')
  

