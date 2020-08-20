%%
clear all;
clc;

inputMC = dlmread('Sweep_n2_t2_d1_b0.100000_J1.000000_B0.000000.dat', ' ', 0, 0);
inputHOTRG = dlmread('HOTRG_Ising_sweep_L2.dat', '\t', 0, 0);
inputATRG = dlmread('Ising_sweep.dat', ' ', 0, 0);

fig = figure;
hold on;
plot(inputMC(:, 1), inputMC(:, 2), '-k', "linewidth", 1.5);
plot(inputHOTRG(:, 1), inputHOTRG(:, 2), ':+', "linewidth", 1.5);
plot(inputATRG(:, 1), inputATRG(:, 2), ':+', "linewidth", 1.5);
plot(inputATRG(:, 1), inputATRG(:, 3), ':+', "linewidth", 1.5);
grid off;
hold off;
set(gca, "linewidth", 1.5, "fontsize", 13)
%axis tight;
axis([0 4 -2.02 -0.4]);
legend('Monte Carlo',
       'HOTRG', 
       'ATRG Impurity',
       'ATRG FD',
       'location', 'northwest');
legend boxoff;
title('Energy of the Ising model, 2D lattice');
xlabel('T');
ylabel('E');

print(fig, 'energy.pdf', '-dpdf', '-landscape', '-bestfit');


fig = figure;
hold on;
semilogy(inputATRG(:, 1), inputATRG(:, 5), '+', "linewidth", 1.5);
semilogy(1.0 ./ (1.0 ./ inputATRG(:, 1) .- 5e-3), inputATRG(:, 6), '+', "linewidth", 1.5);
semilogy(1.0 ./ (1.0 ./ inputATRG(:, 1) .+ 5e-3), inputATRG(:, 7), '+', "linewidth", 1.5);
grid off;
hold off;
set(gca, "linewidth", 1.5, "fontsize", 13)
axis tight;
%axis([0 4 -2.02 -0.4]);
legend('log(Z)',
       'log(Z)_m', 
       'log(Z)_p',
       'location', 'northeast');
legend boxoff;
title('log(Z) of the Ising model, 2D lattice');
xlabel('T');
ylabel('log(Z)');

print(fig, 'logZ.pdf', '-dpdf', '-landscape', '-bestfit');


fig = figure;
hold on;
plot(inputHOTRG(:, 1), inputHOTRG(:, 3), ':+', "linewidth", 1.5);
plot(inputATRG(:, 1), inputATRG(:, 4), ':+', "linewidth", 1.5);
grid off;
hold off;
set(gca, "linewidth", 1.5, "fontsize", 13)
%axis tight;
axis([0 4 -0.02 2]);
legend('HOTRG', 
       'ATRG FD',
       'location', 'northwest');
legend boxoff;
title('Specific Heat of the Ising model, 2D lattice');
xlabel('T');
ylabel('\chi_E');

print(fig, 'specific_heat.pdf', '-dpdf', '-landscape', '-bestfit');