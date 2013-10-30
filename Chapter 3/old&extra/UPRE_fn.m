function[UPREalpha] = UPRE_fn(alpha,b,c,params,Bmult_fn,sigma)

a_params               = params.a_params;
a_s_hat                = a_params.a_s_hat;
[nx2,ny2]              = size(a_s_hat); nx = nx2/2; ny = ny2/2;
a_params.alpha         = alpha;
params.a_params        = a_params;
params.precond_params  = 1./(abs(a_s_hat).^2+alpha);
xalpha                 = CG(zeros(nx,ny),c,params,Bmult_fn);
Axalpha                = Amult_Dirichlet(xalpha,a_s_hat);

% Randomized trace estimation
v                      = 2*(rand(nx,ny)>0.5)-1;
Atv                    = Amult_Dirichlet(v,conj(a_s_hat));
Aalphav                = CG(zeros(nx,ny),Atv,params,Bmult_fn);
trAAalpha              = sum(sum(v.*Amult_Dirichlet(Aalphav,a_s_hat)));

% Evaluate UPRE function.
UPREalpha              = norm(Axalpha(:)-b(:))^2+2*sigma^2*trAAalpha;
