%TwoVarTest.m

% Define the matrix A using the SVD
v1=[1/sqrt(2) 1/sqrt(2)]';
v2=[-1/sqrt(2) 1/sqrt(2)]';
s1 = 1; s2 = 1e-2;
A=s1*v1*v1'+s2*v2*v2';

% Define true x and noise free b
x=[1,1]';
b_e = A*x;

% Create realization from the data model
nsamp = 1000;
sigma = 0.1;
b_samp = repmat(b_e,1,nsamp)+sigma*randn(2,nsamp);

figure(1),
  plot(b_samp(1,:),b_samp(2,:),'k*')
% Create the corresponding least squares solutions 
x_LS = (A'*A)\(A'*b_samp);
figure(2)
  plot(x_LS(1,:),x_LS(2,:),'k*'), hold on,
figure(3)  
  subplot(2,1,1), hist(v1'*x_LS,100), title('v_1^Tx_{LS}')
  subplot(2,1,2), hist(v2'*x_LS,100), title('v_1^Tx_{LS}')
  
