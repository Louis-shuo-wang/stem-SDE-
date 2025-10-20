% pde_to_ode_reduction.m
% PDE -> ODE reduction verification (deterministic)
% - Finite-volume upwind for transport + reaction sources
% - Compare (P_PDE, W_PDE) to ODE solution (P_ODE, W_ODE)
% - L2 relative error over [0,T_end], refine dx, dt (CFL)
% - Check nonnegativity and observe first-order convergence

clear; close all; clc;

%% Parameters (example; replace with your maps if desired)
params = initializeParams;

% initial totals
P0 = 5; W0 = 3;

% initial spatial distributions (exponential tails)
S_profile = @(x) exp(-x);      % normalized to integrate to P0
T_profile = @(x) exp(-0.8*x);  % normalized to integrate to W0

%% Refinement study settings
Nx_list = [50, 100, 200, 400];      % successively refined meshes
errors = zeros(length(Nx_list),1);
errors_P = zeros(length(Nx_list),1);
errors_W = zeros(length(Nx_list),1);
dxs = zeros(length(Nx_list),1);

%% Precompute high-accuracy ODE solution
ode_rhs = @(t, y) ode_solver(t,y,params);
[t_ode_ref, y_ode_ref] = ode45(ode_rhs, linspace(0, params.Tend, 5000), [P0; W0]);
P_ode_ref_fun = @(t) interp1(t_ode_ref, y_ode_ref(:,1), t, 'pchip');
W_ode_ref_fun = @(t) interp1(t_ode_ref, y_ode_ref(:,2), t, 'pchip');

%% Font and style settings (journal-ready)
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
set(groot,'defaultAxesColor','w');
set(groot,'defaultTextColor','k');
set(groot,'defaultAxesXColor','k');
set(groot,'defaultAxesYColor','k');

%% Loop over meshes
for k = 1:length(Nx_list)
    Nx = Nx_list(k);
    dx = params.xmax / Nx;
    dxs(k) = dx;
    x = (0:dx:params.xmax);
    
    % Velocities
    vP = params.vP;
    vW = params.vW;
    
    % CFL timestep
    vmax = max(abs([vP, vW]));
    dt = params.CFL * dx / max(1e-12, vmax);
    Nt = ceil(params.Tend / dt);
    dt = params.Tend / Nt;
    time = (0:Nt)'*dt;
    
    % Initial conditions
    S = S_profile(x);  S = S / (sum(S)*dx) * P0;
    T = T_profile(x);  T = T / (sum(T)*dx) * W0;
    
    P_pde = zeros(Nt+1,1); W_pde = zeros(Nt+1,1);
    P_pde(1) = sum(S)*dx;  W_pde(1) = sum(T)*dx;
    
    % --- Time stepping (finite-volume upwind + reactions)
    for n = 1:Nt
        barP = sum(S)*dx; barW = sum(T)*dx;
        p1 = params.p1(barW);
        p2 = params.p2(barW);
        lambdaP = params.lambdaP(barW);
        lambdaR = params.lambdaR(barP);
        delta = params.delta;
        
        fluxP = zeros(Nx+2,1);
        fluxW = zeros(Nx+2,1);
        
        % Boundary inflow
        fluxP(1) = (1+p1-p2)*lambdaP*barP;
        fluxW(1) = (1-p1+p2)*lambdaP*barP;
        
        % Upwind transport
        if vP >= 0
            fluxP(2:Nx+1) = vP * S(1:Nx);
            fluxP(Nx+2)   = vP * S(Nx+1);
        else
            fluxP(2:Nx+1) = vP * S(2:Nx+1);
        end
        if vW >= 0
            fluxW(2:Nx+1) = vW * T(1:Nx);
            fluxW(Nx+2)   = vW * T(Nx+1);
        else
            fluxW(2:Nx+1) = vW * T(2:Nx+1);
        end
        
        divFluxP = (fluxP(2:Nx+2) - fluxP(1:Nx+1)) / dx;
        divFluxW = (fluxW(2:Nx+2) - fluxW(1:Nx+1)) / dx;
        
        rhsS = -divFluxP' - lambdaP*S + lambdaR*T;
        rhsT = -divFluxW' - (delta + lambdaR)*T;
        
        S = max(S + dt*rhsS, 0);
        T = max(T + dt*rhsT, 0);
        
        P_pde(n+1) = sum(S)*dx;
        W_pde(n+1) = sum(T)*dx;
    end
    
    % --- Compute L2 errors
    P_ode_sample = P_ode_ref_fun(time);
    W_ode_sample = W_ode_ref_fun(time);
    
    errP_L2 = sqrt(trapz(time, (P_pde - P_ode_sample).^2 ));
    errW_L2 = sqrt(trapz(time, (W_pde - W_ode_sample).^2 ));
    errors_P(k) = errP_L2;
    errors_W(k) = errW_L2;
    errors(k) = sqrt(errP_L2^2 + errW_L2^2);
    
    fprintf('Nx=%d, dx=%.4f, dt=%.4e, Nt=%d: RelErrP=%.3e, RelErrW=%.3e, combined=%.3e\n', ...
        Nx, dx, dt, Nt, errP_L2, errW_L2, errors(k));
    
    % Plot only for finest mesh
    if k == length(Nx_list)
        idx_mark = 1:4:length(time);
        p_est = polyfit(log(dxs), log(errors), 1);
        slope = p_est(1);

        % === Composite Figure: FIG1 ===
        figure('Color','w','Units','centimeters','Position',[3 3 18 18]);
        tiledlayout(3,1,'Padding','compact','TileSpacing','compact');
        
        % (a) PDE vs ODE totals
        nexttile;
        plot(time, P_pde, 'b-', 'DisplayName','P PDE'); hold on;
        plot(time, W_pde, 'r-', 'DisplayName','W PDE');
        plot(time(idx_mark), P_ode_sample(idx_mark), 'bo--', 'MarkerSize',5, 'DisplayName','P ODE');
        plot(time(idx_mark), W_ode_sample(idx_mark), 'rs--', 'MarkerSize',5, 'DisplayName','W ODE');
        xlabel('$t$'); ylabel('Totals');
        title('(a) PDE vs ODE (finest mesh)');
        legend('Location','best'); grid on;
        
        % (b) Separate component errors
        nexttile;
        loglog(dxs, errors_P, 's-', 'DisplayName','P error'); hold on;
        loglog(dxs, errors_W, 'd-', 'DisplayName','W error');
        ref_x = [min(dxs), max(dxs)];
        ref_y2 = errors_P(end) * (ref_x / ref_x(end));
        loglog(ref_x, ref_y2, 'k--', 'DisplayName','$O(\Delta x)$');
        text(2.2*ref_x(1), 2*ref_y2(1), '$O(\Delta x)$', 'Color','k');
        xlabel('$\Delta x$'); ylabel('L2 error');
        title('(b) Component errors');
        legend('Location','best'); grid on;
        
        % (c) Combined error and slope
        nexttile;
        loglog(dxs, errors, 'o-', 'DisplayName','Combined error'); hold on;
        ref_y = errors(end) * (ref_x / ref_x(end));
        loglog(ref_x, ref_y, 'k--', 'DisplayName','$O(\Delta x)$');
        text(2.2*ref_x(1), 2*ref_y(1), '$O(\Delta x)$', 'Color','k');
        xlabel('$\Delta x$'); ylabel('Combined L2 error');
        title('(c) Mesh refinement');
        legend('Location','best'); grid on;

        % === Export high-resolution composite figure ===
        exportgraphics(gcf, 'FIG2.eps', 'Resolution', 650);
        disp('FIG1 exported successfully (650 dpi, composite of 3 subplots).');
    end
end

%% Helper function: ODE backbone
function dy = ode_solver(~, y, params)
P = y(1); W = y(2);
p1 = params.p1(W);
p2 = params.p2(W);
lambdaP = params.lambdaP(W);
lambdaR = params.lambdaR(P);
delta = params.delta;
dP = (p1 - p2)*lambdaP*P + lambdaR * W;
dW = (1 - p1 + p2) * lambdaP*P - (delta + lambdaR) * W;
dy = [dP; dW];
end
