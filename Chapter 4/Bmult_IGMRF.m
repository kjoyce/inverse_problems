  function y_array = Bmult_Dirichlet(x_array,params);

%
%
%  Compute array(y), where y = (T'*T + alpha*L)*x. 

  ahat       = params.ahat;
  alpha      = params.alpha;
  Dv         = params.Dv;
  Dh         = params.Dh;
  lhat       = params.lhat;
  Lambda     = params.Lambda;

  [n,n]    = size(x_array);
  TtTx_array = Amult(Amult(x_array,ahat),conj(ahat));
  reg_term   = Dh'*(Lambda.*(Dh*x_array(:))) + Dv'*(Lambda.*(Dv*x_array(:)));
  y_array    = TtTx_array + alpha*reshape(reg_term,n,n);
  
  