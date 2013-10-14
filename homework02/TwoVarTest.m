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
  xx_LS = repmat(x,1,nsamp) + sigma*inv(A'*A)*A'*randn(2,nsamp);
  plot(xx_LS(1,:),xx_LS(2,:),'rx')                               
  legend('Least Squares Solutions','Sampling from Least Squares Distribution')
figure(3)  
  subplot(2,1,1), hist(v1'*x_LS,100); title('v_1^Tx_{LS}')
  subplot(2,1,2), hist(v2'*x_LS,100); title('v_1^Tx_{LS}')

mean(x_LS')
mean(xx_LS')  

norm_curve = @(x,sig,mu)( 1/(sqrt(2*pi)*sig)*exp(-(x-mu).^2/(2*sig^2)) );
figure(4)  
  subplot(2,1,1), [n1,c1] = hist(v1'*x_LS,100);
  bar(c1,n1/sum(n1*(c1(2)-c1(1)))); % Normalized Histogram 
  hold on;
  t = get(gca(),'xlim')*[1-(0:.01:1);0:.01:1]; % This makes a linspace from the current axes' xlim
  plot(t,norm_curve(t,(sigma/s1),v1'*x),'r-'); % Plot normal curve
  title('v_1^Tx_{LS}')

  subplot(2,1,2), [n2,c2] = hist(v2'*x_LS,100);
  bar(c2,n2/sum(n2*(c2(2)-c2(1)))), % Normalized Histogram 
  hold on;
  t = get(gca(),'xlim')*[1-(0:.01:1);0:.01:1]; % This makes a linspace from the current axes' xlim
  plot(t,norm_curve(t,(sigma/s2),v2'*x),'r-'); % Plot normal curve
  title('v_2^Tx_{LS}')

  set(gcf, 'PaperPosition', [0 0 10 10]); %Position plot at left hand corner with width 5 and height 5.
  set(gcf, 'PaperSize', [10 10]); %Set the paper to have width 5 and height 5.
  saveas(figure(4),'prob_density.pdf')
