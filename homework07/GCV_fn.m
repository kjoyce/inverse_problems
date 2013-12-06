function[G] = GCV_fn(alpha,A,L,b) 

Reg_mat = A*((A'*A+alpha*L)\A');
n = length(b);
G = n*norm(Reg_mat*b-b)^2/(n-trace(Reg_mat))^2;