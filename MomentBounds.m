% Empirical verification of theoretical moment bounds for the stochastic
% system, used to plot figure 3

function MomentBounds()

clear; close all; clc;
rng(1);

%% --- Parameters and feedback maps ---
params = initializeParams;
delta = params.delta;
p1fun = params.p1;
p2fun = params.p2;
lambdaPfun = params.lambdaP;
lambdaRfun = params.lambdaR;

%% --- SDE setup ---
T_end  = 40;
dt     = 1e-3;
dt_out = 0.05;
tgrid  = 0:dt_out:T_end;
Nt     = numel(tgrid);
Nmc    = 500; 
p_list = [1 2 3];
z0 = [1;1];   

%% --- Monte Carlo simulations ---
SumMC_all = zeros(Nt,length(p_list), Nmc);
Sum2MC_all = zeros(Nt,length(p_list),Nmc);

parfor imc = 1:Nmc

    P = z0(1); W = z0(2);
    t = 0; next_out = 1;
    out = zeros(Nt,2); out(1,:) = [P,W];

    for n=1:round(T_end/dt)
        p1 = p1fun(W);
        p2 = p2fun(W);
        lambdaP = lambdaPfun(W);
        lambdaR = lambdaRfun(P);

        % drift terms
        muP = (p1-p2)*lambdaP*P + lambdaR*W;
        muW = (1-p1+p2)*lambdaP*P - (delta+lambdaR)*W;

        % diagonal diffusion terms
        sigP = [sqrt(p1*lambdaP*P), -sqrt(p2*lambdaP*P),  0,                         sqrt(lambdaR*W),  0];
        sigW = [0                 , 2*sqrt(p2*lambdaP*P), sqrt((1-p1-p2)*lambdaP*P), -sqrt(lambdaR*W), -sqrt(delta*W)];

        % Euler–Maruyama update
        dB = sqrt(dt)*randn(5,1);
        P = max(P + muP*dt + sigP*dB, 0);
        W = max(W + muW*dt + sigW*dB, 0);

        % record on coarse grid
        t = t + dt;
        if next_out <= Nt && mod(n,round(dt_out/dt)) == 0
            out(next_out,:) = [P,W];
            next_out = next_out + 1;
        end
    end
    out(end, :) = [P,W];
    % accumulate moments
    localSum = zeros(Nt, length(p_list));
    localSum2 = zeros(Nt, length(p_list));
    for ip=1:length(p_list)
        p = p_list(ip);
        M = out(:,1).^p + out(:,2).^p;
        localSum(:,ip) = M;
        localSum2(:,ip) = M.^2;
    end

    SumMC_all(:,:,imc) = localSum;
    Sum2MC_all(:,:,imc)= localSum2;
end

SumMC = sum(SumMC_all, 3);
Sum2MC = sum(Sum2MC_all, 3);

% average
Emc = SumMC / Nmc;
StdMC = sqrt((Sum2MC - Nmc*Emc.^2) / (Nmc-1));

%% --- Exponential fit log(E[P^p+W^p]) ≈ log C + p K t ---
t_fit_mask = tgrid >= 5;
Khat = zeros(size(p_list)); Chat = zeros(size(p_list));
for ip=1:length(p_list)
    y = log(max(Emc(:,ip),1e-12));
    pf = polyfit(tgrid(t_fit_mask), y(t_fit_mask),1);
    Khat(ip) = pf(1) / p_list(ip);
    Chat(ip) = exp(pf(2));
end
fprintf('Fitted K per p: '); fprintf(' %.4f', Khat); fprintf('\n');


set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultAxesFontSize',12);
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLineLineWidth',1.2);
set(groot,'defaultAxesLineWidth',1.2);
set(groot,'defaultAxesTickDir','out');
set(groot,'defaultAxesBox','off');
set(groot,'defaultAxesColor','w');        % white background
set(groot,'defaultTextColor','k');        % black text
set(groot,'defaultAxesXColor','k');       % black axes
set(groot,'defaultAxesYColor','k');       % black axes

%% --- Plot moments and fitted curves ---
figure('Color','w','Units','centimeters','Position',[3 3 17.4 13]); 
cols = lines(length(p_list));

for ip = 1:length(p_list)
    subplot(length(p_list),1,ip); hold on;

    mu  = Emc(:,ip);
    sig = StdMC(:,ip);

    % --- mean ± 1.96 std (≈95% CI) shading ---
    x = tgrid(:)';  % ensure row vector
    lower = (mu - 1.96*sig)';  % row
    upper = (mu + 1.96*sig)';  % row
    fill_x = [x, fliplr(x)];
    fill_y = [lower, fliplr(upper)];
    fill(fill_x, fill_y, cols(ip,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');

    % --- mean line ---
    plot(tgrid, mu, 'Color', cols(ip,:), 'LineWidth', 1.4);

    % --- exponential fit ---
    plot(tgrid, Chat(ip)*exp(p_list(ip)*Khat(ip)*tgrid), 'k--', 'LineWidth', 1.2);

    grid on;
    ylabel(sprintf('$E[P^{%d}+W^{%d}]$', p_list(ip), p_list(ip)), 'Interpreter', 'latex');
    legend({'mean $\pm 95\%$ CI','mean','fit'}, 'Location', 'northwest');

    if ip == 1
        title('Monte-Carlo mean $\pm 95\%$ CI and fitted $C e^{p K t}$', 'Interpreter', 'latex');
    end
    if ip == length(p_list)
        xlabel('time');
    end
end
end