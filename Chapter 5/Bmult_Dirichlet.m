  function y_array = a_mult_toep(x_array,params);

%
%
%  Compute array(y), where y = (T'*T + alpha*L)*x. T is assumed to be
%  block Toeplitz with Toeplitz blocks (BTTB). Block circulant
%  extensions are used to compute matrix-vector products involving T
%  and T'. The products have O(n log n) complexity through the use of
%  2-D FFT's. 

  t_ext_hat = params.t_ext_hat;
  L         = params.L;
  lambda    = params.lambda;
  delta     = params.delta;
  
  %  Compute lambda T'*(T*x) + delta L*x via circulant extension.
  
  [n2x,n2y] = size(t_ext_hat);
  TtTx_array = Amult_Dirichlet(Amult_Dirichlet(x_array,t_ext_hat),conj(t_ext_hat));
  Lx_array   = reshape(L*x_array(:),n2x/2,n2y/2); 
  y_array    = lambda*TtTx_array + delta* Lx_array;
  
  