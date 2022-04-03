function [model] = BCPF_ghd(Y, varargin)
%  Bayesian CP Factorization of a Complete Tensor
%
%  [model] = BCPF(Y, 'PARAM1', val1, 'PARAM2', val2, ...)
%
%  INPUTS
%     Y              - input tensor
%     'init'         - Initialization method
%                     - 'ml'  : SVD initilization (default)
%                     - 'rand': Random matrices
%     'maxRank'      - The initialization of rank (larger than true rank)
%     'dimRed'       - 1: Remove unnecessary components automaticly (default)
%                    - 0: Not remove
%     'maxiters'     - max number of iterations (default: 100)
%     'tol'          - lower band change tolerance for convergence dection
%                      (default: 1e-5)
%     'noise'        - whether noise is updated
%                        - 'on': update noise parameter (default)
%                        - 'off': fixed noise parameter (1e-5)
%     'verbose'      - visualization of results
%                       - 0: no
%                       - 1: text (default)
%   OUTPUTS
%      model         - Model parameters and hyperparameters
%
%   Example:
%
%        [model] = BCPF(Y, 'init', 'ml', 'maxRank', 10, 'dimRed', 1, 'maxiters', 100, ...
%                                'tol', 1e-6, 'verbose', 3);
%
% < Bayesian CP Factorization >
% Copyright (C) 2020 Zhongtao Chen and Lei Cheng
% Acknowledgement
% This code is built based on the Matlab Code from the following paper:
% Q. Zhao, L. Zhang, and A. Cichocki, "Bayesian CP factorization of incomplete 
% tensors with automatic rank determination," IEEE Transactions on 
% Pattern Analysis and Machine Intelligence, vol. 37, no. 9, pp. 1751-1763, Sep. 2015.
% We highly appreciated the authors of this paper. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Set parameters from input or by using defaults
dimY = size(Y);
N = ndims(Y);

ip = inputParser;
ip.addParameter('init', 'ml', @(x) (ismember(x,{'ml','rand'})));
ip.addParameter('maxRank', max(dimY), @isscalar);
ip.addParameter('dimRed', 1, @isscalar);
ip.addParameter('maxiters', 100, @isscalar);
ip.addParameter('tol', 1e-5, @isscalar);
ip.addParameter('noise', 'on', @(x)ismember(x,{'on','off'}));
ip.addParameter('predVar', 0, @isscalar);
ip.addParameter('verbose', 1, @isscalar);
ip.parse(varargin{:});

init  = ip.Results.init;
R   = ip.Results.maxRank;
maxiters  = ip.Results.maxiters;
tol   = ip.Results.tol;
verbose  = ip.Results.verbose;
DIMRED   = ip.Results.dimRed;
noise = ip.Results.noise;

%% Initialization
Y = tensor(Y);
nObs = prod(dimY);

if  strcmp(noise,'on')
    a_beta0      = 1e-6;
    b_beta0      = 1e-6;
else
    a_beta0      = 1e-1;
    b_beta0      = 1e-6;
end

dscale = std(Y(:))/N;
Y = Y./dscale;
gammas = ones(R,1);
beta = 1;

switch init
    case 'ml'    % Maximum likelihood
        Z = cell(N,1);
        ZSigma = cell(N,1);
        for n = 1:N
            ZSigma{n} = eye(R);
            [U, S, V] = svd(double(tenmat(Y,n)), 'econ');
            if R <= size(U,2)
                Z{n} = U(:,1:R)*(S(1:R,1:R)).^(0.5);
            else
                Z{n} = [U*(S.^(0.5)) randn(dimY(n), R-size(U,2))];
            end
        end
    case 'rand'   % Random initialization
        Z = cell(N,1);
        ZSigma = cell(N,1);
        for n = 1:N
            ZSigma{n} = eye(R);
            Z{n} = randn(dimY(n),R);
        end
        %       X = tensor(ktensor(Z));
        %       dscale = std(Y(:))/std(X(:));
end

X = double(ktensor(Z));

% --------- E(aa') = cov(a,a) + E(a)E(a')----------------
EZZT = cell(N,1);
for n=1:N
    EZZT{n} = Z{n}'*Z{n} + dimY(n)*ZSigma{n};
end

Fit =0;

%% init_value for GHB prior
z_g = gammas;
inv_z_g = gammas;

scale_coeff_a = 1;
scale_coeff_b = 0;
scale_coeff_lambda = - min(dimY);%max(dimY);

k_a = -scale_coeff_lambda/2 + 1;
theta_a = 1e-6;

z_lambda0 = scale_coeff_lambda*ones(R,1);
z_a0 = scale_coeff_a*ones(R,1);
z_b0 = scale_coeff_b*ones(R,1);
a_mackay = ones(R,1);

%% Model learning
for it=1:maxiters
    %% Update factor matrices
    Aw = diag(gammas);
    for n=1:N
        % compute E(Z_{\n}^{T} Z_{\n})
        ENZZT = ones(R,R);
        for m=[1:n-1, n+1:N]
            ENZZT =  ENZZT.*EZZT{m};
        end
        % compute E(Z_{\n})
        FslashY = double(khatrirao_fast(Z{[1:n-1, n+1:N]},'r')' * tenmat(Y, n)');
        ZSigma{n} = (beta * ENZZT + Aw)^(-1);
        Z{n} = (beta * ZSigma{n} * FslashY)';
        EZZT{n} = Z{n}'*Z{n} + dimY(n)*ZSigma{n};
    end
    
    %% Update latent tensor X
    epsilon_stop = 1e-10; % this value can be set to trade-off the convergence speed and learning accuracies 
    diff=conj(double(ktensor(Z))) - X;
    if norm(diff(:),'fro')<=  epsilon_stop *prod(dimY)  %
        disp('\\\======= Converged===========\\\');
        break;
    end
    X = double(ktensor(Z));
  
    %% Update hyperparameters gamma via GHB
    % update the mackay's hyper parameters z
    z_lambda = z_lambda0 - (sum(dimY)/2)*ones(R,1);
    b_add = 0;
    for n = 1 : N
        b_add = b_add +  diag(Z{n}'*Z{n}) + dimY(n)*diag(ZSigma{n});
    end     
    z_b = z_b0 + b_add;
    %z_a = z_a0;
    z_a = a_mackay;
    z_g = zeros(1,R);
    for i = 1 : R
        if abs(z_b(i))>= 1e-6      
            z_g(i) = (sqrt(z_a(i))./sqrt(z_b(i)+eps)).*besselk(z_lambda(i)-1,sqrt(z_a(i)).*sqrt(z_b(i))+eps)./besselk(z_lambda(i),sqrt(z_a(i)).*sqrt(z_b(i))+eps);
            inv_z_g(i) = (sqrt(z_b(i))./sqrt(z_a(i))).*besselk(z_lambda(i)+1,sqrt(z_a(i)).*sqrt(z_b(i))+eps)./besselk(z_lambda(i),sqrt(z_a(i)).*sqrt(z_b(i))+eps);
        end
        if abs(z_b(i))<= 1e-6 || isnan(z_g(i))
            z_g(i) =  (sqrt(z_a(i)))./(sqrt(z_b(i))+eps);
            inv_z_g(i) = (sqrt(z_b(i) + eps)./sqrt(z_a(i)));
        end
    end
    gammas = z_g;

    for i = 1 : R
        a_mackay(i) = (k_a+z_lambda0(i)/2)./(theta_a + inv_z_g(i)/2);
    end
    
    
    %% update noise beta
    EX2 = ones(R,R);
    for n=1:N
        EX2 = EX2.*EZZT{n};
    end
    EX2 = sum(EX2(:));
    err = Y(:)'*Y(:) - 2*Y(:)'*X(:) + EX2;
    if  strcmp(noise,'on')
        a_betaN = a_beta0 + 0.5*nObs;
        b_betaN = b_beta0 + 0.5*err;
    else
        a_betaN = a_beta0;
        b_betaN = b_beta0;
    end
    beta = a_betaN/b_betaN;
    Fit = 1 - sqrt(sum(err(:)))/norm(Y(:));
    
    %% Prune out unnecessary components
    Zall = cell2mat(Z);
    comPower = diag(Zall' * Zall);
    comTol = sum(dimY)*eps(norm(Zall,'fro'));
    rankest = sum(comPower> comTol );
    if max(rankest)==0
        disp('Rank becomes 0 !!!');
        break;
    end
    if DIMRED==1  && it >=2
        if R~= max(rankest)
            indices = comPower > comTol;
            gammas = gammas(indices);
            for n=1:N
                Z{n} = Z{n}(:,indices);
                ZSigma{n} = ZSigma{n}(indices,indices);
                EZZT{n} = EZZT{n}(indices, indices);
            end
            z_lambda0 = z_lambda0(indices,1);
            a_mackay = a_mackay(indices, 1);
            z_a0 = z_a0(indices,1);
            z_b0 = z_b0(indices,1);
            R = max(rankest);
        end
    end
    
    %% Display progress
    if verbose
        fprintf('Iter. %d: Fit = %g, R = %d \n', it, Fit, rankest);
    end

end

%% Prepare the results
SNR = 10*log10(var(X(:))*beta);
X = ktensor(Z)*dscale;
X = arrange(X);

%% Output
model.Z = Z;
model.X = X;
model.ZSigma = ZSigma;
model.gammas = gammas;
model.Fit = Fit;
model.SNR = SNR;
model.iternum = it;
model.TrueRank = rankest;
