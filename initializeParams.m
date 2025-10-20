function params = initializeParams()

params.xmax = 20;
params.Tend = 10;
params.CFL = 0.45;         % CFL < 1 for explicit upwind
params.vP = 0.1;   % advection speed for S
params.vW = 0.12;   % advection speed for T
params.delta = 0.5;

% Feedback maps (examples)
p1max = 0.60; k1 = 0.50;
p2max = 0.30; k2 = 0.20;
params.p1 = @(W) p1max ./ (1 + k1*W);
params.p2 = @(W) p2max ./ (1 + k2*W);

params.lambdaP0 = 1.00; beta = 0.20;      % division decreases with W
params.lambdaP = @(W) params.lambdaP0 ./ (1 + beta*W);

params.lambdaR0 = 0.50; gamma = 0.10;     % dedifferentiation saturating
params.lambdaR = @(P) params.lambdaR0 ./ (1 + gamma*P);