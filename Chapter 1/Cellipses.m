clear all, close all
% Enter the covariance matrix.
C = [2 1; 1 4];
R = chol(C);
N = 10000;
y = R'*randn(2,N);

% Compute the chi^2-test
for i = 1:N
  xtinvCx(i) = norm(R'\y(:,i))^2;
end
% Computation of 95% and 99% confidence limits using Chi^2 with 
% N degrees of freedom. Check to see if draws match these values.
clevel_95 = chi2inv(.95,2); 
clevel_99 = chi2inv(.99,2); 
fprintf('Percentage of draws within the 95 percent confindence limits = %2.5f\n',100*sum(xtinvCx<=clevel_95)/N);
fprintf('Percentage of draws within the 99 percent confindence limits = %2.5f\n',100*sum(xtinvCx<=clevel_99)/N);

% Plot draws from the distribution together with the 95 and 99% level curves.
figure(1)
 plot(y(1,:),y(2,:),'*')
 xlabel('y_1')
 ylabel('y_2')
 % Now plot the level curves.
 [X,Y]=meshgrid(-10:0.1:10,-10:0.1:10);
 A = inv(C);
 Z=A(1,1)*X.^2 + (A(1,2)+A(2,1))*X.*Y + A(2,2)*Y.^2;
 hold on
  contour(X,Y,Z,clevel_95,'LineWidth',2)
  contour(X,Y,Z,clevel_99,'LineWidth',2)
 hold off
 legend('Draws from N(0,C)','95% Confindence Level Curve','99% Confindence Level Curve','Location','Southeast')
