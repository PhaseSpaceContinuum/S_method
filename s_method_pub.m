function [res, f, t] = s_method_pub(y, mode, winlenT, winlenF, params)
    % function [res, f, t] = s_method_pub(y, mode, winlenT, winlenF, params)
    %   Implementation of multiple variants of the S-Method time-frequency representation, 
    %   which is a refinement of the Spectrogram. It's TF resolution is similar to that of the Smoothed Pseudo Wigner-Ville Distribution.
    %
    % INPUT:
    %   y@numeric(1,N) or numeric(N,1)...   1D input signal
    %   mode@numeric(1)...                  S-method mode:
    %                                           1: Standard implementation with Hann time-window and Hann frequency window
    %                                               "A method for time-frequency analysis", L. Stankovic, IEEE Transactions on Signal Processing, 1994, 42, p.225-229
    %                                           2: Adaptive frequency window length / Adaptive, element-wise thresholding 
    %                                               "Digital Signal Processing with selected topics", L. Stankovic, 2015, p. 649
    %                                           3: Multiple time-window method with properly optimized weights
    %                                               "Multiwindow S-method for instantaneous frequency estimation and its application
    %                                                in radar signal analysis", Orovic, Stankovic et al., 10.1049/iet-spr.2009.0059
    %                                          -3: Multiple time-window method with static weights - optimized for constant amplitude per TF
    %                                                window
    %                                           4: 2+3
    %                                           0: Raw spectrogram
    %                                       optional; default: mode = 2
    %   winlenT@numeric(1)...               integer time-window length
    %   winlenF@numeric(1)...               integer frequency-window length
    %   params@struct(1)...
    %   params.P@numeric(1)...              if mode=3|4|5; number of orthogonal polynomials
    %                                       optional; default: P = 5
    %   params.thres@numeric(1)...          if mode=2|4; threshold of max(abs(spectrogram)) in percent
    %                                       optional; default: thres = 1
    %   params.tmax@numeric(1)...           if mode=3|4|5; maximal range of coordinate of time-window. Determines sampling of Hermitean time-windows.
    %                                       optional; default: tmax = 6
    %
    % OUTPUT:
    %  if nargout == 0:
    %   plot tfr, similar to pspectrum(...,'spectrogram')
    %  otherwise:
    %   res@numeric(Omega,N2)...            real-valued S-Method data, Omega == #Freq. Bins = min(8*winlenT, 1024) 
    %                                                                  N2    == #Time Samples is given by possible overlaps on signal
    %   f@numeric(N,1)...                   sampled frequencies as normalized frequencies in units of radian per sample
    %   t@numeric(N,1)...                   sampled time-instances in units of 1
    %   
    % All implementations use MATLAB's internal STFT functionality, an FFTLength of min(8*winlenT, 1024), and an overlapLength of
    % min(0.75*N, winlenT-1).
    %
    % Test via:
    %   a=exp(2*pi*1i*100*linspace(-1,1,1024).^5);
    %   N = 4*512; b = exp(1i*N/3*cos(1*linspace(0,2*pi,N)));
    %   figure(1), subplot(2,3,1), pspectrum(a, 'spectrogram'); subplot(2,3,2), imagesc(s_method(a, 1, 64, 15)); subplot(2,3,3), imagesc(s_method(a, 2, 64, 15))
    %              subplot(2,3,4), pspectrum(b, 'spectrogram'); subplot(2,3,5), imagesc(s_method(b)), subplot(2,3,6), imagesc(s_method(b, 2))
    %
    %
    %   % Smaller interferences mode 1->2
    %   figure(2)
    %   sig2 = (2+sin(linspace(-64,64,128))).*exp(1i*64*linspace(-1,1,128).^2);
    %   subplot(1,3,1), s_method(sig2, 1), title('S-Method Mode: 1')
    %   subplot(1,3,2), s_method(sig2, 2), title('S-Method Mode: 2')
    %   subplot(1,3,3), s_method(sig2, 3), title('S-Method Mode: 3')
    %
    %   figure(3)
    %   N = 2048; sig2 = (2+sin(linspace(-64,64,N))).*exp(1i*N/2*cos(linspace(-1,1,N).^2));
    %   subplot(2,2,1), s_method(sig2, 1), ax1 = gca; ax1.CLim = [0.85, 1]*ax1.CLim(2); title('S-Method Mode: 1')
    %   subplot(2,2,2), s_method(sig2, 2), ax2 = gca; ax2.CLim = [0.85, 1]*ax2.CLim(2); title('S-Method Mode: 2')
    %   subplot(2,2,3), s_method(sig2, 3), ax3 = gca; ax3.CLim = [0.85, 1]*ax3.CLim(2); title('S-Method Mode: 3')
    %   subplot(2,2,4), sp(sig2), ax4 = gca; ax4.CLim = [-50, ax4.CLim(2)]; title('Spectrogram')
    %
    % V. 2.03
    % 05.05.2020, Tobias Birnbaum
    %
    % Requires: Signal Processing Toolbox for the STFT evaluation and Matlab R2019a or later.
    %
    % License: http://creativecommons.org/licenses/by-nc-sa/4.0/, Creative Commons License
    % This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License 
    % (CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/).
    
    
    %% Initialization BEGIN
    nel = numel(y);
    overlapLength = floor(0.75*nel);


    if(nargin < 2), mode = 2; end
    if(nargin < 3 || isempty(winlenT)), winlenT = min(floor(nel/4), 256); end
    if(nargin < 4), winlenF = 5; end % Make small
    winlenF = winlenF + mod(winlenF+1, 2); % Ensure odd winlen
    
    %% Parse and set "P"
    use_P = (mode == -3 || mode == 3 || mode == 4);
    if((use_P && nargin < 5) || (use_P && nargin >= 5 && ~isfield(params, 'P')))
        params.P = 5; % Default argument
    end
    if(use_P)
        P = min(max([params.P(:); 0]), 7); % Only const. weights until 7 implemented yet.
    end
    
    %% Parse and set "thres"
    use_thres = (mode == 2 || mode == 4);
    if((use_thres && nargin < 5) || (use_thres && nargin >= 5 && ~isfield(params, 'thres')))
        params.thres = 1;
    end
    if(use_thres)
        threshPerc = min([max([params.thres(:); 0]); 100]);
    end

    %% Parse and set "tmax"
    use_tmax = use_P;
    if((use_tmax && nargin < 5) || (use_tmax && nargin >= 5 && ~isfield(params, 'tmax')))
        params.tmax = 6; %10;
    end
    if(use_tmax)
        tmax = max([params.tmax(:); 0.1]);
    end

    
    y = y(:);
    
    winT = hann(winlenT); %kaiser(winlenT, 0.85);

    % Needs to be full window with central element at w(winlenF+1)
    winF = hann(2*winlenF+1);
    
    nel = numel(y);
    if(nel < winlenT)
        y = borderpad(y, [winlenT, 1]);
        nel = numel(y);
    end
    
    overlapLength = min(overlapLength, winlenT-1);
    numFreqBinRequest = winlenT;
    Omega = numFreqBinRequest;
    Nt = nel - overlapLength;
    
    % Initialization END
    
    
    if(mode == 0 || mode == 1 || mode == 2)
        [F, f, ~] = stft(y,'Window',winT, 'FFTLength', numFreqBinRequest, 'OverlapLength', overlapLength);
        F = F.';

        if(use_thres), thres = threshPerc/100 * max(abs(F(:))).^2; end
    end
    
    switch(mode)
        case 0 % Raw spectrogram
            res = abs(F).^2;
        case 1 % naive
            % Use recursive formula
            % Work time + frequency -parallel
            
            % 0.th term
            res = abs(F).^2 * winF(winlenF+1); % == F.*conj(F) * winF(winlenF+1)
            
            for k = 1:Omega
                for l = 1:winlenF
                    if k+l <= Omega && k-l > 0
                        res(:, k) = res(:, k) + (winF(winlenF+1-l)+winF(winlenF+1+l))*real(F(:, k+l).*conj(F(:, k-l))); % real to suppress imag-floating point errors 
                    end
                end
            end
        case 2 % Adaptive frequency window length / Adaptive, element-wise thresholding
            % Use recursive formula
            % Work time + frequency -parallel
            
            % 0.th term
            res = abs(F).^2 * winF(winlenF+1); % == F.*conj(F) * winF(winlenF+1)
            for k = 1:Omega
                for l = 1:winlenF
                    if k+l <= Omega && k-l > 0
                        update = real(F(:, k+l).*conj(F(:, k-l))); % real to suppress imag-floating point errors 
                        update(abs(update) < thres) = 0;
                        if(nnz(update) == 0), continue, end % Early exit
                        res(:, k) = res(:, k) + (winF(winlenF+1-l)+winF(winlenF+1+l))*update; 
                    end
                end
            end
        case 3 % Multiple time-windows - optimized weights
            % Use recursive formula
            % Work time-parallel
            res = zeros(Nt, Omega, class(y));
            wCurr = calculateOptWeights(y, winlenT, P, tmax); % This step is somewhat computationally expensive.
            
            for deg = [1:P]
                winT = hermiteanWindow(winlenT, deg-1, tmax);
                [F, ~, t] = stft(y,'Window',winT, 'FFTLength', numFreqBinRequest, 'OverlapLength', overlapLength);
                F = F.'; 
                
                w = reshape(wCurr(t, deg), Nt,1); % Use equivalence: diag(a)*b == a(:).*b
                
                % 0.th term
                res = res + (w * winF(winlenF+1)) .* abs(F).^2;
                
                for k = 1:Omega
                    for l = 1:winlenF
                        if k+l <= Omega && k-l > 0
                            res(:, k) = res(:, k) + (w * winF(winlenF+1-l) + w * winF(winlenF+1+l)) .* (real(F(:, k+l).*conj(F(:, k-l)))); % real to suppress imag-floating point errors 
                        end
                    end
                end
            end
            [~, f, ~] = stft(y,'Window',winT, 'FFTLength', numFreqBinRequest, 'OverlapLength', 0);
            
            
        case -3 % Multiple time-windows const. amplitude weights
            % Use recursive formula
            % Work time-parallel
            res = zeros(Nt, Omega, class(y));
            wCurr = calculateOptWeights(ones(size(y), class(y)), winlenT, P, tmax).'; % This step is somewhat computationally expensive.
            wCurr = wCurr(1:P);
            
            for deg = [1:P]
                w = wCurr(deg);
                
                winT = hermiteanWindow(winlenT, deg-1, tmax);
                F = stft(y,'Window',winT, 'FFTLength', numFreqBinRequest, 'OverlapLength', overlapLength).';
                
                % 0.th term
                res = res + w * abs(F).^2 * winF(winlenF+1);
                
                for k = 1:Omega
                    for l = 1:winlenF
                        if k+l <= Omega && k-l > 0
                            res(:, k) = res(:, k) + (w * winF(winlenF+1-l) + w * winF(winlenF+1+l)) * real(F(:, k+l).*conj(F(:, k-l))); % real to suppress imag-floating point errors 
                        end
                    end
                end
            end
            [~, f, ~] = stft(y,'Window',winT, 'FFTLength', numFreqBinRequest, 'OverlapLength', 0);
            
        case 4 % 2+3
            % Use recursive formula
            % Work time-parallel
            res = zeros(Nt, Omega, class(y));
            wCurr = calculateOptWeights(y, winlenT, P, tmax);
            
            for deg = [1:P]
                winT = hermiteanWindow(winlenT, deg-1, tmax);
                [F, ~, t] = stft(y,'Window',winT, 'FFTLength', numFreqBinRequest, 'OverlapLength', overlapLength);
                F = F.';
                
                thres = threshPerc/100 * max(abs(F(:))).^2;
                
                w = reshape(wCurr(t, deg), Nt, 1); % Use equivalence: diag(a)*b == a(:).*b
                
                % 0.th term
                res = res + (w * winF(winlenF+1)) .* abs(F).^2;
                
                for k = 1:Omega
                    for l = 1:winlenF
                        if k+l <= Omega && k-l > 0
                            update = real(F(:, k+l).*conj(F(:, k-l))); % real to suppress imag-floating point errors 
                            update(abs(update) < thres) = 0;
                            if(nnz(update) == 0), continue, end % Early exit
                            res(:, k) = res(:, k) + (w * winF(winlenF+1-l) + w * winF(winlenF+1+l)).*update; 
                        end
                    end
                end
            end
            [~, f, ~] = stft(y,'Window',winT, 'FFTLength', numFreqBinRequest, 'OverlapLength', 0);
        otherwise
            error('s_method:invalid_mode', 'Supported modes {1,..,4}');
    end
    clear update;
    res = res.';
    t = linspace(1, nel, Nt);
    
    if(nargout == 0)
        % Resize isotropically to 2k for plotting, in case output is too high-res
        if(numel(Omega*Nt)>2048^2)
            fac = max(2048/[Omega, Nt]);
            res = imresize(abs(res), fac, 'bicubic');
        end
        pltH = imagesc(20*log10(abs(res)));
        ax = gca;
        ax.XTickLabel = {min(t), floor(mean(t)), max(t)};
        d = min(diff(f));
        ax.YTickLabel = {round(min(f-d)/pi, 2), round(mean(f)/pi, 1), round(max(f)/pi, 2)};
        
        ax.XTick = [min(ax.XLim), mean(ax.XLim), max(ax.XLim)];
        ax.YTick = [min(ax.YLim), mean(ax.YLim), max(ax.YLim)];
        pltH.Parent.YDir = 'normal';
        ax.XLabel.String = "Samples";
        ax.YLabel.String = "Normalized Frequency (\times \pi radians)";
        colorbar;
        
        clear res f t;
    end
end


%% Auxiliary functions
function win = hermiteanWindow(winlen, deg, tmax)
    % function win = hermiteanWindow(winlen, deg, tmax)
    %   Returns a Hermitean window of length winlen, evaluated on 
    %       t = linspace(-winlen/2,winlen/2-1,winlen)/winlen*tmax.
    %
    % INPUT:
    %   winlen@numeric(1)...    window length
    %   deg@numeric(1)...       natural number [0,....]; degree of the hermitean polynomial to use
    %   tmax@numeric(1)...      numeric scale of the window; determines how much of the polynomials will be sampled
    %
    % OUTPUT:
    %   win@numeric(winlen, 1)...  hermitean polynomial window, with center at floor(winlen/2)+1
    %
    % 14.07.2019
    % V. 1.00, Tobias Birnbaum
    
    if(deg < 0 || round(deg) ~= deg), error('hermiteanWindow:too_small_degree', 'The degree needs to be a natural number >=0.'), end
    
    
    % Coordinate range choice via "tmax" determines resolution
    t = linspace(-floor(winlen/2),ceil(winlen/2)-1,winlen).'/winlen*tmax;
    
    switch(deg)
        case 0
            win = pi^(-0.25) * exp(-t.^2/2);
        case 1
            win = sqrt(2)*pi^(-0.25) * t .* exp(-t.^2/2);
        otherwise
            win = sqrt(2/deg) * t .* hermiteanWindow(winlen, deg-1, tmax) - sqrt((deg-1)/deg) * hermiteanWindow(winlen, deg-2, tmax);
    end
end

function weights = calculateOptWeights(sig, winlen, P, tmax)
    % function weights = calculateOptWeights(sig, winlen, P, tmax)
    %   Calculates optimal weights for all Hermitean polynomials of degree p \in [0, P-1] per sample point of the signal.
    %   In general required to do so only for all sampled windows.
    %
    %   Solve the follwoing linear system:
    %   For all p=0:P-1 hw(:, p) = hermiteanWindow(winlen, p);
    %   y_ii = sum_p=0^(P-1) d(p, t) * (sum_m=(-T/2)^(T/2-1) sig(t+m)^2 * hw(m, p)^2 * m^ii) 
    %                               / (sum_m=(-T/2)^(T/2-1) sig(t+m)^2 * hw(m, p)^2)
    %
    %   y_0 == 1; y_ii == 0, with i>0
    %   d \in R^(P x numel(sig) in this algorithm. d will be returned as d^T!
    %
    % pp1 will be p+1!
    % hw2 will be hw^2!
    %
    %   PBC will be used on "sig" while evaluating d.
    %
    %   Solve y = M * d for each time instant.
    %   M will require P^2*T of temporary memory. 
    %
    % See: "Multiwindow S-method for instantaneous frequency estimation and its application
    %       in radar signal analysis", Orovic, Stankovic et al., 10.1049/iet-spr.2009.0059
    %
    % INPUT:
    %   sig@numeric(T,1)...     complex-valued input signal -> only absolute value will be used in processing
    %   winlen@numeric(1)...    window-length for which weights should be calculated
    %   P@numeric(1)...         number of Hermitean polynomials to use in multi-window approach
    %   tmax@numeric(1)...      numeric scale of the window; determines how much of the polynomials will be sampled
    %                           Must match "tmax" in computation of Hermitean polynomials in the algorithm.
    %
    % OUTPUT:
    %   weights@numeric(T, P)...optimal or const.-amplitude weights for T windows of length(winlen), centered at t\[1,T] on the signal
    %                           for all p\in[0,P-1] Hermitean polynomials
    %                           Usually: weights will be subsampled in T, at the time-instants the STFT was evaluated on
    %
    % 14.07.2019
    % V. 1.00, Tobias Birnbaum
    
    sig = abs(sig).^2;
    T = numel(sig);
    
    %% Using PBC
    sig = [sig(end-floor(winlen/2)+1:end); sig; sig(1:ceil(winlen/2)-1)];
    
    % List of pre-computed weights for constant amplitude signals for P = 1..7
    weights_const_amplitude_list = {[1], [1.5, -0.5], [1.75, -1, 0.25], [1.875, -1.375, 0.625, -0.125], ...
        [1.9375, -1.625, 1, -0.375, 0.0625], [1.96875, -1.78125, 1.3125, -0.6875, 0.21875, -0.03125], ...
        [1.984375, -1.875, 1.546875, -1, 0.453125, -0.125, 0.015625]};
    weights_const_amplitude = weights_const_amplitude_list{P};
    
    %% Precompute all window functions
    hw2 = zeros(winlen, P);
    for p = 0:P-1
        hw2(:, p+1) = hermiteanWindow(winlen, p, tmax).^2;
    end
    if(~isa(hw2, class(sig)))
        hw2 = cast(hw2, class(sig));
    end
    
    %% Assemble matrix M
    % Dimensions: [rows, cols, slides] = [ii, p, n] = [window id on signal, polynom order, time]
    % 1st row is 1 in every slide.
    % Assume the same number of windows (used on the signal), as polynoms used -> M is square + invertible
    
    M = zeros([P, P, T], class(sig));
    M(1, :, :) = 1;
    
    m = linspace(-floor(winlen/2),ceil(winlen/2)-1,winlen).'/winlen*tmax; % Coordinate choice must match coordinate choice in window construction
    
    M = permute(M, [3, 1, 2]); % temporariy shape [n, ii, p] = [time, window id, polynom order]
    for p = 0:P-1
        % sig, hw2, are already squared
        denomi = conv(sig, hw2(:, p+1), 'valid');
        denomi(denomi == 0) = 1;
        for ii = 1:P-1
            enumerat = conv(sig, hw2(:, p+1).*m.^ii, 'valid');
            M(:, ii+1, p+1) = enumerat ./ denomi;
        end
    end
    clear sig denomi enumerat hw2 m;
    M = permute(M, [2, 3, 1]); % revert to original shape [ii, p, n] = [window id, polynom order, time]
    
    %% Work around floating point arithmetic errors - divisions by 0
    M(abs(M)<1e-7) = 0;
    M(isnan(M) | isinf(M)) = 0;
    
    
    %% Solve T linear systems for P coefficients each
    weights = zeros(P, T);
    y = [1; zeros([P-1, 1])];
    for t = 1:T
        if(cond(M(:, :, t)) < 1e8)  % If M is well conditioned solve LS
            weights(:, t) = M(:, :, t) \ y;
        else                        % fall back to constant-amplitude weights, if M is ill-conditioned 
                                    %   (such as in const. amplitude scenario)
            weights(:, t) = weights_const_amplitude(:);
        end
    end
    
    weights = weights.';
end


function Y = borderpad(X, S, val)
    % function Y = borderpad(X, S, val)
    %
    %   adds black borders to image X until such that it has dimensions S
    %
    %   v1.10
    %   11.02.2020, Tobias Birnbaum

    if(nargin < 3), val = 0; end
    
    s = size(X);
    s(1:numel(S)) = S;

    if(val == 0)
        Y = zeros(s);
    else
        Y = val*ones(s);
    end
    xs = floor((s(2)-size(X,2))/2);
    if S(1) == 1 
        ys =0;
    else
    ys = floor((s(1)-size(X,1))/2);
    end
    Y(ys+1:ys+size(X,1), xs+1:xs+size(X,2), :) = X;
end

