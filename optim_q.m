function [optq, optcost] = optim_q(r, alpha, delta, D, k)

    warning('off', 'manopt:getHessian:approx')
    
    warning('off')
    
    % Create the problem structure.
    n = size(r,1);
    manifold = sympositivedefinitesimplexfactory(n, k);
    problem.M = manifold;

    % Define the problem cost function and its Euclidean gradient.
    problem.cost  = @(q) f(q, r, alpha, delta, D);
    problem.egrad = @(q) egradf(q, r, alpha, delta, D);      % notice the 'e' in 'egrad' for Euclidean


    % Numerically check gradient consistency (optional).
    %checkgradient(problem);

    % Solve
    options.verbosity = 0;
    options.maxiter = 500;
    
    %c = 0.05;
   
    %x0 = cat(3, c * eye(n), (1-c) * eye(n)); 
    
    [x, xcost, info, options] = trustregions(problem, [], options);
    
    optq = x(:,:,1);
    
    optcost = xcost;

end

function [out] = f(q, r, alpha, delta, D)

    s = size(r,1);
      
    out = - trace(r * q(:,:,1)) / 2; % an interaction part
    
    out = out - delta / 2 / D * trace( diag([alpha, 1-alpha]) * (eye(s) - q(:,:,1)) ); % tr( diag(alpha,1-alpha) * (I-q) ) part
    
    out = out - delta / 2 * ( alpha * log( det( eye(s) + ( eye(s) - q(:,:,1) ) * diag([1, 0]) / D ) ) + (1-alpha) * log( det( eye(s) + ( eye(s) - q(:,:,1) ) * diag([0, 1]) / D ) ) );  % a logdet part
        
    B1tr = (eye(s) - q(:,:,1)) * diag([1,0]) / D + eye(s);      % expe[ (Tr[B] - 2) * (2 - Tr[B^{-1}]) ] part
    
    B0tr = (eye(s) - q(:,:,1)) * diag([0,1]) / D + eye(s);
    
    % Tr[B^{-1}] = s - 1/D * Tr[ (I-q) * diag(eta, 1-eta) * (I + 1/D * (I-q) * diag(eta, 1-eta))^{-1} ] 
    
    tr_Bm1_1 = s - trace( (eye(s) - q(:,:,1)) * diag([1, 0]) * inv(B1tr) ) / D;
    
    tr_Bm1_0 = s - trace( (eye(s) - q(:,:,1)) * diag([0, 1]) * inv(B0tr) ) / D;
    
    out = out + delta / 2 * alpha * (trace(B1tr) - 2) * ( 2 - tr_Bm1_1 );
    
    out = out + delta / 2 * (1-alpha) * (trace(B0tr) - 2) * ( 2 - tr_Bm1_0 );
    
    out = out + delta / 2 * alpha * (s - tr_Bm1_1);
    
    out = out + delta / 2 * (1-alpha) * (s - tr_Bm1_0);
    
end

function [out] = egradf(q, r, alpha, delta, D)
     
    s = size(r,1);
    
    out = cat(3, - r/2, zeros(s));   % der of -1/2 * tr(rq)
    
    out = out + delta / 2 / D * cat(3, diag([alpha, 1-alpha]), zeros(s));  % der of -delta / 2 / D * tr( diag(alpha, 1-alpha) * (I-q) ) 
    
    B1temp = eye(s) + diag([1, 0]) * (eye(s) - q(:,:,1)) / D;            % der of a logdet part
    
    T1 = delta * alpha * inv(B1temp) * diag([1, 0]) / D;
    
    out(:, :, 1) = out(:, :, 1) + (T1' + T1)/2 / 2;
    
    B0temp = eye(s) + diag([0, 1])* (eye(s) - q(:,:,1)) / D;
    
    T1 = delta * (1-alpha) * inv(B0temp) * diag([0, 1]) / D;
    
    out(:, :, 1) = out(:, :, 1) + (T1' + T1)/2 / 2;
    
    B1tr = eye(s) + (eye(s) - q(:,:,1)) * diag([1,0]) / D;      % der of an expe[ (Tr[B] - 2) * (2 - Tr[B^{-1}]) ] part
    
    B0tr = eye(s) + (eye(s) - q(:,:,1)) * diag([0,1]) / D;
    
    der_TrB_1 = - diag([1, 0]) / D;
    
    der_TrB_0 = - diag([0, 1]) / D;
    
    % out(:,:,1) = out(:,:,1) + delta * (alpha * der_TrB_1  + (1-alpha) * der_TrB_0); % -> der. of Tr[B]
    
    temp1 = eye(s) + diag([1, 0]) * (eye(s) - q(:, :, 1)) / D;
    
    temp0 = eye(s) + diag([0, 1]) * (eye(s) - q(:, :, 1)) / D;
    
    eta = 0;    % -> der. of Tr[B^{-1}]
    
    A = diag([eta, 1 - eta]);
    
    T1 = eye(s) - q(:,:,1);
    
    T2 = inv(eye(s) + A * T1 / D);
    
    T3 = T2 * A;
    
    der_TrBm1_0 = - ( (T2 * A * T1 * T3 / D - T3)' + (T3 * T1 * T2 * A / D - T3) ) / 2 / D;
    
    tr_Bm1_0 = s - trace( (eye(s) - q(:,:,1)) * diag([0, 1]) * inv(B0tr) ) / D;
    
    eta = 1;
    
    A = diag([eta, 1 - eta]);
    
    T1 = eye(s) - q(:,:,1);
    
    T2 = inv(eye(s) + A * T1 / D );
    
    T3 = T2 * A;
    
    der_TrBm1_1 = - ( (T2 * A * T1 * T3 / D - T3)' + (T3 * T1 * T2 * A / D - T3) ) / 2 / D;
    
    tr_Bm1_1 = s - trace( (eye(s) - q(:,:,1)) * diag([1, 0]) * inv(B1tr) ) / D;
       
    % der of expe[ (Tr[B] - 2) * (2 - Tr[B^{-1}]) ] for eta = 1: 
    out(:,:,1) = out(:,:,1) + delta / 2 * alpha * ( - (trace(B1tr) - 2) * der_TrBm1_1 + der_TrB_1 * ( 2 - tr_Bm1_1 ) );
    
    % der of expe[ (Tr[B] - 2) * (2 - Tr[B^{-1}]) ] for eta = 1: 
    out(:,:,1) = out(:,:,1) + delta / 2 * (1-alpha) * ( - (trace(B0tr) - 2) * der_TrBm1_0 + der_TrB_0 * ( 2 - tr_Bm1_0 ) );
    
    out(:,:,1) = out(:,:,1) + alpha * delta / 2 * (-1) * der_TrBm1_1;
    
    out(:,:,1) = out(:,:,1) + (1-alpha) * delta / 2 * (-1) * der_TrBm1_0;
    
end