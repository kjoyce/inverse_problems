function[GCValpha] = GCV_fn(alpha,b,c,params,Bmult_fn)

[n,n]                  = size(b);
a_params               = params.a_params;
ahat                   = a_params.ahat;
lhat                   = a_params.lhat;
a_params.alpha         = alpha;
params.a_params        = a_params;
params.precond_params  = 1./(abs(ahat).^2+alpha*lhat);
xalpha                 = CG(zeros(n,n),c,params,Bmult_fn);
Axalpha                = Amult(xalpha,ahat);

% Randomized trace estimation
v                      = 2*(rand(n,n)>0.5)-1;
Atv                    = Amult(v,conj(ahat));
Aalphav                = CG(zeros(n,n),Atv,params,Bmult_fn);
trAAalpha              = sum(sum(v.*Amult(Aalphav,ahat)));

% Evaluate GCV function.
GCValpha               = n^2*norm(Axalpha(:)-b(:))^2./(n^2-trAAalpha)^2;