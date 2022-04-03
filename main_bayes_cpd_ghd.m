clc;
clear all;
close all;
randn('state',1); rand('state',1); %#ok<RAND>
addpath(genpath(pwd));

%% Generate a low-rank tensor
DIM = [30,30,30];  % Dimensions of data          
R = 18;  % True CP rank
Z = cell(length(DIM),1);
for m=1:length(DIM)
    Z{m} =  gaussSample(zeros(R,1), eye(R), DIM(m));
end
X = double(ktensor(Z)); % Generate tensor by factor matrices

results = zeros(9,5);
rtt = 0;
%% Add noise
for SNR = 10 % Noise levels
    sigma2 = var(X(:))*(1/(10^(SNR/10)));
    mont_max = 10;
    rse_list = zeros(1, mont_max);
    rmse_list = zeros(1, mont_max);
    time_list = zeros(1, mont_max);
    rank_list = zeros(1, mont_max);
    
    for mont_num = 1 : mont_max
        randn('state',mont_num); rand('state',mont_num); %#ok<RAND>
        GN = sqrt(sigma2)*randn(DIM);
        
        %% Generate observation tensor Y
        Y = X + GN;
        
        %% Run PCPD-GH
        ts = tic;
        [model] = BCPF_ghd(Y, 'init', 'ml', 'maxRank', 2*max([DIM]), 'dimRed', 1, 'tol', 1e-10, 'maxiters', 200, 'verbose', 0);
        t_total = toc(ts);
        
        %% Performance evaluation
        X_hat = double(model.X);
        err = X_hat(:) - X(:);
        rmse = sqrt(mean(err.^2));
        rmse_list(mont_num) = rmse;
        rrse = sqrt(sum(err.^2)/sum(X(:).^2));
        rse_list(mont_num) = rrse;
        rank_list(mont_num) = model.TrueRank;
        time_list(mont_num) =  t_total;
        
        %% Report results
        fprintf('\n------------Bayesian CP Factorization in %d mont run----------------------------------------------------------------\n', mont_num)
        fprintf('SNR = %g, True Rank=%d\n', SNR, R);
        fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, \nEstimated SNR = %g, Time = %g\n', ...
            rrse, rmse, model.TrueRank, model.SNR, t_total);
        fprintf('--------------------------------------------------------------------------------------------------------------------------\n')

    end

fprintf('\n-------------Final Results--------------------------------------------------------------\n')
fprintf('Observation ratio = %g, SNR = %g, True Rank = %d\n', 1, SNR, R);
fprintf('Mean RMSE = %g, Mean Estimated Rank = %g, Std Estimated Rank = %g, Detection Accuracy = %g, Time = %g\n', ...
    mean(rmse_list), mean(rank_list), std(rank_list), 1-sum(rank_list~=R)/mont_max, mean(time_list));
fprintf('--------------------------------------------------------------------------------------------------------------------------\n')

rtt = rtt + 1;
results(rtt,1) = mean(rmse_list);
results(rtt,2) = mean(rank_list);
results(rtt,3) = std(rank_list);
results(rtt,4) = 1-sum(rank_list~=R)/mont_max;
results(rtt,5) = mean(time_list);

end
