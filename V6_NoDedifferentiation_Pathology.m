function V6_NoDedifferentiation_Pathology()
% V6_NoDedifferentiation_Pathology
% Demonstrates the "no–dedifferentiation" pathology (Proposition E):
%   (i) Subcritical extinction when p1 < p2
%   (ii) Supercritical moment blow–up when p1 > p2

clear; close all; clc;
rng(1);

%% --- Global parameters -------------------------------------------------
delta       = 0;                % death rate
lambdaR     = 0.0;            % no dedifferentiation
T_end       = 50;                  % final simulation time
dt          = 1e-3;                % integration step
dt_out      = 0.05;                % output step
t_out       = (0:dt_out:T_end).';  % output grid
Nmc         = 500;                 % number of trajectories
z0          = [1; 1];              % initial condition


%% (i) Subcritical case: extinction when p1 < p2
fprintf('\n=== (i) Subcritical case ===\n');
p1       = 0.435;
p2       = 0.5;
lambdaP  = 1.0;

[Pmat_sub, ~] = simulate_SDE_ensemble(z0, T_end, dt, dt_out, Nmc, ...
    p1, p2, lambdaP, lambdaR, delta);

% Ensemble mean of P(t)
P_mean = mean(Pmat_sub, 2);

% Fit log(mean(P)) for exponential decay
% mask = t_out >= fit_t0;
% pf = polyfit(t_out(mask), log(P_mean(mask)), 1);
% slope_est = pf(1); intercept_est = pf(2);

% fprintf('Fitted exponential decay rate ≈ %.4f (expected −%.4f)\n', ...
%     slope_est, gamma_theory);

% --- Plot: subcritical extinction ---
setPlotDefaults;
figure('Color','w','Units','centimeters','Position',[3 3 17.4 13]);
semilogy(t_out(1:end-1), P_mean(1:end-1), 'b-'); hold on;
% semilogy(t_out, exp(intercept_est + slope_est*t_out), 'k--', 'LineWidth', 1.2);
xlabel('Time'); ylabel('$E[P_t]$', 'Interpreter','latex');
title(sprintf('(i) Subcritical: empirical $E[P]$'));
grid on;


%% (ii) Supercritical case: blow–up when p1 > p2

fprintf('\n=== (ii) Supercritical case ===\n');
p1       = 0.5;
p2       = 0.435;
lambdaP  = 1.0;
gamma_theory = 0.1;
p_list = [1 2];   % moment orders

[Pmat_sup, ~] = simulate_SDE_ensemble(z0, T_end, dt, dt_out, Nmc, ...
    p1, p2, lambdaP, lambdaR, delta);

Khat = zeros(numel(p_list),1);
Chat = zeros(numel(p_list),1);
EM   = zeros(numel(t_out), numel(p_list));

for ip = 1:numel(p_list)
    p = p_list(ip);
    EM(:,ip) = mean(Pmat_sup.^p, 2);
    y = log(max(EM(:,ip), 1e-16));
    pf = polyfit(t_out, y, 1);
    Khat(ip) = pf(1)/p; 
    Chat(ip) = exp(pf(2));
end

fprintf('Fitted K per p: ');
fprintf('p=%d→K=%.4g; ', [p_list; Khat.']);
fprintf('\nExpected growth κ ≥ %.4g\n', gamma_theory);

% --- Plot: supercritical blow-up ---
cols = lines(numel(p_list));
figure('Color','w','Units','centimeters','Position',[3 3 17.4 13]);
for ip = 1:numel(p_list)
    subplot(numel(p_list),1,ip); hold on;
    p = p_list(ip);
    plot(t_out(1:end-1), EM(1:end-1,ip), 'Color', cols(ip,:));
    plot(t_out(1:end-1), Chat(ip)*exp(p*Khat(ip)*t_out(1:end-1)), 'k--');
    ylabel(sprintf('$E[P^{%d}]$', p));
    legend({'Monte–Carlo mean','Fitted $Ce^{pK t}$'}, 'Location','northwest');
    if ip == 1
        title('(ii) Supercritical: empirical $E[P^p]$ vs fitted exponential');
    end
    if ip == numel(p_list)
        xlabel('Time');
    end
    grid on;
end

% fprintf('\nSummary:\n (i) Subcritical slope ≈ %.4g (expected −%.4g)\n', ...
%     slope_est, gamma_theory);
fprintf(' (ii) Supercritical K per p: '); fprintf('%.4f ', Khat); fprintf('\n');

end


%% simulate_SDE_ensemble — full 2×5 diffusion, parallel trajectories

function [Pmat, Wmat] = simulate_SDE_ensemble(z0, T_end, dt, dt_out, Nmc, ...
    p1, p2, lambdaP, lambdaR, delta)
% Returns:
%   Pmat, Wmat: (nTime × Nmc) ensemble trajectories at discrete output times.

t_out = (0:dt_out:T_end).';
nSamp = numel(t_out);
Nt_full = round(T_end/dt);
samp_idx = round(t_out/dt) + 1;

Pmat = zeros(nSamp, Nmc);
Wmat = zeros(nSamp, Nmc);

parfor imc = 1:Nmc
    P = z0(1); W = z0(2);
    out = zeros(nSamp,2);
    samp_i = 1;
    out(1,:) = [P, W];

    for k = 2:Nt_full

        % Drift
        muP = (p1 - p2)*lambdaP*P + lambdaR*W;
        muW = (1 - p1 + p2)*lambdaP*P - (delta + lambdaR)*W;

        % Full 2×5 diffusion coefficients (channels: SR, SD, ASD, R, D)
        sigP = [ sqrt(p1*lambdaP*P), ...
                -sqrt(p2*lambdaP*P), 0, 0, 0 ];
        sigW = [ 0, ...
                 2*sqrt(p2*lambdaP*P), ...
                 sqrt((1-p1-p2)*lambdaP*P), ...
                 0, -sqrt(delta*W) ];

        % Euler–Maruyama step
        dB = sqrt(dt)*randn(5,1);
        P = max(P + muP*dt + sigP*dB, 0);
        W = max(W + muW*dt + sigW*dB, 0);

        % Record on output grid
        if samp_i < nSamp && k == samp_idx(samp_i+1)
            samp_i = samp_i + 1;
            out(samp_i,:) = [P, W];
        end
    end
    Pmat(:,imc) = out(:,1);
    Wmat(:,imc) = out(:,2);
end
end
