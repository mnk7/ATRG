%%
clear all;
clc;

inputHOTRG = dlmread('HOTRG_Ising_sweep.dat', '\t', 0, 0);
inputATRG = dlmread('Ising_sweep.dat', ' ', 0, 0);

fig = figure;
hold on;
plot(inputHOTRG(:, 1), inputHOTRG(:, 2), ':+', "linewidth", 1.5);
plot(inputATRG(:, 1), inputATRG(:, 2), ':+', "linewidth", 1.5);
grid off;
hold off;
set(gca, "linewidth", 1.5, "fontsize", 13)
axis ([0 4 -2 -0.5]);
legend('HOTRG', 
       'ATRG',
       'location', 'southeast');
legend boxoff;
title('Energy of the Ising model, 32x32 lattice');
xlabel('T');
ylabel('E');

print(fig, 'energy.pdf', '-dpdf', '-landscape', '-bestfit');