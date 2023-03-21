function optrq = optim_r(alpha, delta, D)

    warning('off', 'manopt:getHessian:approx')
    
    warning('off')
    
    % Create the problem structure.
    n = 2;
    k = 2;
    manifold = sympositivedefinitefactory(n);
    problem.M = manifold;

    % Define the problem cost function and its Euclidean gradient.
    problem.cost  = @(r) (-1) * fr(r, alpha, delta, D);

    % Solve.
    
    options = struct();
    options.maxiter = 100;
    
    [x, xcost, info, options] = neldermead(problem, [], options);
    
    optrq = struct();
    
    optrq.r = x;
       
    optrq.q = optim_q(optrq.r, alpha, delta, D, k);
    
end


function [out] = fr(r, alpha, delta, D)

    r;

    s = size(r,1);
    
    k = 2;
      
    [optimq, optcost] = optim_q(r, alpha, delta, D, k);
    
    %optimq = eye(2);
    
    %out = - trace(r * optimq) / 2; % an interaction part
    
    out = optcost;
    
    out = out - log(det(r + eye(s))) / 2; % a logdet part
    
    out = out + trace(r * inv(r + eye(s)) * r) / 2;
    
    out = out + trace(r * inv(r + eye(s)) ) / 2;
        
end