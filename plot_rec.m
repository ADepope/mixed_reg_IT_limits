deltas = 0.2:0.1:0.9;
%deltas = 20;

D = 1e-5;

alphas = 0.6:0.09:1;
alphas = 0.6;

acor = zeros( length(deltas), length(alphas) );


for i = 1:length(deltas)
    
    for j = 1:length(alphas)
        
        i
        j
   
        optrq = optim_r(alphas(j), deltas(i), D);

        qopt = optrq.q;

        acor(i,j) = qopt(2,2) % (delta, alpha)
        
    end
    
end

acor