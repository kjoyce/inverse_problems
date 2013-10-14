%TwodTest.m

% Define the matrix A using the SVD
v1=[1/sqrt(2) 1/sqrt(2)]';
v2=[-1/sqrt(2) 1/sqrt(2)]';
s1 = 1; s2 = 1e-2;
A=s1*v1*v1'+s2*v2*v2';

% Define true x and noise free b
x=[1,1]'
b_e = A*x;
b = b_e + [0.026,0.075]';

% Least squares solution.
x_LS = (A'*A)\(A'*b)

% TSVD solution
A_TSVD = s1*v1*v1';
x_TSVD = s1\(v1'*b)*v1
