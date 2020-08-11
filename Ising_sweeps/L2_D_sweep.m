%%
clear all;
clc;

D2 = dlmread('L2_D2.dat', ' ', 0, 0);
D3 = dlmread('L2_D3.dat', ' ', 0, 0);
D4 = dlmread('L2_D4.dat', ' ', 0, 0);
D8 = dlmread('L2_D8.dat', ' ', 0, 0);
D16 = dlmread('L2_D16.dat', ' ', 0, 0);


fig = figure;
hold on;
plots(1) = semilogy(D2(:, 1), D2(:, 5), '+', "linewidth", 1.5);
set(gca,'ColorOrderIndex', 1)
semilogy(1.0 ./ (1.0 ./ D2(:, 1) .- 5e-3), D2(:, 6), '+', "linewidth", 1.5);
set(gca,'ColorOrderIndex', 1)
semilogy(1.0 ./ (1.0 ./ D2(:, 1) .+ 5e-3), D2(:, 7), '+', "linewidth", 1.5);

set(gca,'ColorOrderIndex', 2)
plots(2) = semilogy(D3(:, 1), D3(:, 5), '+', "linewidth", 1.5);
set(gca,'ColorOrderIndex', 2)
semilogy(1.0 ./ (1.0 ./ D3(:, 1) .- 5e-3), D3(:, 6), '+', "linewidth", 1.5);
set(gca,'ColorOrderIndex', 2)
semilogy(1.0 ./ (1.0 ./ D3(:, 1) .+ 5e-3), D3(:, 7), '+', "linewidth", 1.5);

set(gca,'ColorOrderIndex', 3)
plots(3) = semilogy(D4(:, 1), D4(:, 5), '+', "linewidth", 1.5);
set(gca,'ColorOrderIndex', 3)
semilogy(1.0 ./ (1.0 ./ D4(:, 1) .- 5e-3), D4(:, 6), '+', "linewidth", 1.5);
set(gca,'ColorOrderIndex', 3)
semilogy(1.0 ./ (1.0 ./ D4(:, 1) .+ 5e-3), D4(:, 7), '+', "linewidth", 1.5);

set(gca,'ColorOrderIndex', 4)
plots(4) = semilogy(D8(:, 1), D8(:, 5), '+', "linewidth", 1.5);
set(gca,'ColorOrderIndex', 4)
semilogy(1.0 ./ (1.0 ./ D8(:, 1) .- 5e-3), D8(:, 6), '+', "linewidth", 1.5);
set(gca,'ColorOrderIndex', 4)
semilogy(1.0 ./ (1.0 ./ D8(:, 1) .+ 5e-3), D8(:, 7), '+', "linewidth", 1.5);

set(gca,'ColorOrderIndex', 5)
plots(5) = semilogy(D16(:, 1), D16(:, 5), '+', "linewidth", 1.5);
set(gca,'ColorOrderIndex', 5)
semilogy(1.0 ./ (1.0 ./ D16(:, 1) .- 5e-3), D16(:, 6), '+', "linewidth", 1.5);
set(gca,'ColorOrderIndex', 5)
semilogy(1.0 ./ (1.0 ./ D16(:, 1) .+ 5e-3), D16(:, 7), '+', "linewidth", 1.5);
grid off;
hold off;
set(gca, "linewidth", 1.5, "fontsize", 13)
%axis tight;
axis([0.1 4 10^-1 20]);
legend(plots, 
       'log(Z) D2',
       'log(Z) D3',
       'log(Z) D4',
       'log(Z) D8',
       'log(Z) D16',
       'location', 'northeast');
legend boxoff;
title('log(Z) of the Ising model, 2D lattice');
xlabel('T');
ylabel('log(Z)');

print(fig, 'L2_D_sweep.pdf', '-dpdf', '-landscape', '-bestfit');