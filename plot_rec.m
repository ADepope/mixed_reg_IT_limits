deltas = 0.5:0.1:2;
deltas = 0.6;

D = 1e-7;

alphas = 0.6:0.09:1;
alphas = 0.9;

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

%acor_flip = flip( acor , 1 ) ;

figure(1)

im = image(alphas, deltas, acor, 'CDataMapping','scaled');

% Create ylabel
ylabel({'deltas',''});

% Create xlabel
xlabel('alphas');


figure(2)

plot(alphas, 1/2./(1-alphas))

% Create ylabel
ylabel({'deltas',''});

% Create xlabel
xlabel('alphas');