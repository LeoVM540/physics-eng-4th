%% Numerical Lab HS1: Lorenz system, Poincare maps and unstable periodic orbits

clear; close all; clc;

sigma = 10;
b     = 8/3;
itmax = 50;

% LaTeX in all plots (labels, titles, legends and axis numbers)
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

%% (a) Equilibria and linear stability

r = 25:-0.05:0;      % we go from r = 25 down to r = 0
N = length(r);

O  = zeros(3,N);     % origin equilibrium
CP = nan(3,N);       % C+ equilibrium
CM = nan(3,N);       % C- equilibrium
gO = zeros(1,N);     % largest real part of the eigenvalues at O
gP = nan(1,N);       % largest real part of the eigenvalues at C+
gM = nan(1,N);       % largest real part of the eigenvalues at C-

% initial guesses at r = 25
uO = [0; 0; 0];
uP = [5; 5; 20];
uM = [-5; -5; 20];

for k = 1:N
    % origin
    uO = newton(uO, r(k), sigma, b, itmax);
    O(:,k) = uO;
    gO(k) = max(real(eig(jac(uO, r(k), sigma, b))));

    % C+ and C- only exist for r > 1
    if r(k) > 1
        uP = newton(uP, r(k), sigma, b, itmax);   % previous solution as initial guess
        uM = newton(uM, r(k), sigma, b, itmax);
        CP(:,k) = uP;
        CM(:,k) = uM;
        gP(k) = max(real(eig(jac(uP, r(k), sigma, b))));
        gM(k) = max(real(eig(jac(uM, r(k), sigma, b))));
    end
end

fprintf('Origin becomes unstable at r = %.2f\n', min(r(gO > 0)));
fprintf('C+ and C- become unstable at r = %.2f\n', min(r(gP > 0)));

% bifurcation diagram: blue = stable, red = unstable
figure; hold on; grid on;
plot(r(gO < 0), O(1, gO < 0), 'b.');
plot(r(gO > 0), O(1, gO > 0), 'r.');
plot(r(gP < 0), CP(1, gP < 0), 'b.');
plot(r(gP > 0), CP(1, gP > 0), 'r.');
plot(r(gM < 0), CM(1, gM < 0), 'b.');
plot(r(gM > 0), CM(1, gM > 0), 'r.');
xlabel('$r$'); ylabel('$x_{eq}$');
title('Equilibria (blue: stable, red: unstable)');

% largest real part of the eigenvalues
figure; hold on; grid on;
plot(r, gO, 'k', r, gP, 'b');
plot(r, 0*r, 'k--');
xlabel('$r$'); ylabel('$\max \, \mathrm{Re}(\lambda)$');
legend('$O$', '$C_\pm$');

%% (b) Phase portrait analysis
% One r value in each stability region found in (a):
%   r = 0.5  : 0 < r < 1          -> O stable node (only equilibrium)
%   r = 1.2  : 1 < r < 1.346      -> O unstable, C+- stable
%   r = 10   : 1.346 < r < 24.74  -> O saddle, C+- stable
%   r = 20   : 1.346 < r < 24.74  -> C+- stable foci (long chaotic transient)
%   r = 25   : r > 24.74          -> O and C+- unstable (chaotic attractor)
y0 = [1; 1; 1];
tspan = [0 50];
R = [0.5 1.2 10 20 25];

for r = R
    [t, y] = ode45(@(t,u) lorenz(u, r, sigma, b), tspan, y0);
    figure; hold on; grid on;
    scatter3(y(:,1), y(:,2), y(:,3), 4, t, 'filled');   % colour = time
    c = colorbar;
    c.Label.String = 'time';
    xlabel('$x$'); ylabel('$y$'); zlabel('$z$');
    title(sprintf('$r = %g$', r));
    view(3);
end

%% (c) Unstable limit cycles (Poincare map + Newton)
% r values of the unstable periodic orbits shown in prac1.pdf
Rc = [22 19.5 17];

for r = Rc
    [t, y] = ode45(@(t,u) lorenz(u, r, sigma, b), tspan, y0);
    figure; hold on; grid on;
    scatter3(y(:,1), y(:,2), y(:,3), 4, t, 'filled');   % colour = time
    c = colorbar;
    c.Label.String = 'time';
    xlabel('$x$'); ylabel('$y$'); zlabel('$z$');
    title(sprintf('$r = %g$', r));
    view(3);
end


%% Functions

function F = lorenz(u, r, sigma, b)
x = u(1); y = u(2); z = u(3);
F = [sigma*(y - x);
     r*x - y - z*x;
     -b*z + x*y];
end

function J = jac(u, r, sigma, b)
x = u(1); y = u(2); z = u(3);
J = [-sigma,  sigma,  0;
      r - z,  -1,    -x;
      y,       x,    -b];
end

function u = newton(u, r, sigma, b, itmax)
for it = 1:itmax
    F = lorenz(u, r, sigma, b);
    if norm(F) < 1e-12
        break
    end
    du = -jac(u, r, sigma, b) \ F;
    u = u + du;
end
end
