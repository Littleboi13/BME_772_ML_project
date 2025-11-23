%% BME 772 Final Project Filtering Code
% Data Truncation -> Filtering -> *Feature Extraction* -> Machine Learning
% Feature Extraction: Time Domain, Frequency Domain
% Student IDs: Jacklyn D'Ascenzo (xxxxx7753), Mya Hao (xxxxx8133), Tanvir Hassan (xxxxx4056), Janneza Macaalay (xxxxx2196)

%% Time Domain Feature Extraction
function [meanEEG, stdEEG, kurtEEG, skewEEG, AR_coeff] = time_domain_features(signal)
    numCh = size(signal, 1); 

    % Preallocate output variables
    meanEEG = zeros(numCh, 1);
    stdEEG = zeros(numCh, 1);
    kurtEEG   = zeros(numCh, 1);
    skewEEG   = zeros(numCh, 1);
    AR_coeff = cell(numCh, 1);
    
    for ch = 1:numCh
        x = signal(ch,:);

        % 1. Mean and Standard Deviation
        meanEEG(ch) = mean(x);
        stdEEG(ch) = std(x);

        % 2. Kurtosis and Skewness
        kurtEEG = kurtosis(x);
        skewEEG = skewness(x);

        % 3. Determine AR order using AIC
        N = length(x);
        pmax = 10;
        AIC = zeros(pmax, 1);
        for p = 1:pmax
            [~, e] = aryule(x, p); % AR model
            AIC(p) = 2*p + N*log(e); % AIC formula
        end
        [~, AR_order_opt] = min(AIC);

        % 4. Compute AR coefficients with optimal order
        [a, ~] = aryule(x, AR_order_opt);
        AR_coeff{ch} = -a(2:end);  % exclude leading 1, store as row vector
    end
end

function [activity, mobility, complexity] = hjorth_parameters(signal)
    numCh = size(signal,1);

    % Preallocate output variables
    activity   = zeros(numCh,1);
    mobility   = zeros(numCh,1);
    complexity = zeros(numCh,1);

    for ch = 1:numCh
        x = signal(ch,:);
        
        dx = diff(x); % first derivative
        ddx = diff(dx); % second derivative

        % Hjorth Activity
        activity(ch) = var(x); % Activity = variance

        % Hjorth Mobility
        mobility(ch) = sqrt(var(dx) / var(x)); % Square root of (variance of first derivative / variance of signal)

        % Hjorth Complexity
        complexity(ch) = sqrt(var(ddx) / var(dx)) / mobility(ch); % Ratio of mobility of derivative to mobility of original
    end
end

%% Frequency Domain Feature Extraction
    % Power of Each Frequency Band
function [alpha_fPower, beta_fPower, gamma_fPower, delta_fPower, theta_fPower] = feature_domain_bandPower(signal, Fs)
    % α = 8–13 Hz; β = 13–30Hz; γ = 30–100Hz; δ = 0.5–4Hz; θ = 4–8Hz;
    signal = double(signal);
    numCh = size(signal, 1);

    % Preallocate output variables
    alpha_fPower = zeros(numCh, 1);
    beta_fPower  = zeros(numCh, 1);
    gamma_fPower = zeros(numCh, 1);
    delta_fPower = zeros(numCh, 1);
    theta_fPower = zeros(numCh, 1);

    for ch = 1:numCh
        x = signal(ch,:);
        alpha_fPower(ch) = bandpower(x, Fs, [8 13]);
        beta_fPower(ch)  = bandpower(x, Fs, [13 30]);
        gamma_fPower(ch) = bandpower(x, Fs, [30 100]);
        delta_fPower(ch) = bandpower(x, Fs, [0.5 4]);
        theta_fPower(ch) = bandpower(x, Fs, [4 8]);
    end
end

    % Ratios Between the Power of Each Frequency Band
function [d_a, d_b, d_g, d_t, t_a, t_b, t_g, a_b, a_g, b_g, b_at, t_ab, dt_ab, low_high, high_low] = compute_bandpower_ratios(alpha_fPower, beta_fPower, gamma_fPower, delta_fPower, theta_fPower)
    % Stability constant
    epsv = 1e-12;

    % Safe log-ratio function (no dividing by zero!)
    safe = @(num, den) log((num + epsv) ./ (den + epsv));

    % Ratios 
    d_a = safe(delta_fPower, alpha_fPower); % δ/α -> Increase is associated with seizures and slowing of EEG
    d_b = safe(delta_fPower, beta_fPower); % δ/β -> Elevated delta relative to beta indicates slowing or ictal transition
    d_g = safe(delta_fPower, gamma_fPower); % δ/γ -> High delta compared to gamma shows pre-ictal suppression of fast activity
    d_t = safe(delta_fPower, theta_fPower); % δ/θ -> Larger delta over theta associated with seizure onset
    
    t_a = safe(theta_fPower, alpha_fPower); % θ/α  -> Drowsiness, reduced awareness/relaxed wakefulness with eyes closed -> α drops and θ rises before a seizure -> increase = sign of seizure
    t_b  = safe(theta_fPower, beta_fPower); % θ/β -> Theta increase versus beta seen in impaired awareness before seizures 
    t_g = safe(theta_fPower, gamma_fPower); % θ/γ -> Higher theta over gamma may predict seizures
    
    a_b = safe(alpha_fPower, beta_fPower); % α/β -> Associated with being awake: changes reflect shifts in attention/arousal, may be disrupted pre-ictally
    a_g = safe(alpha_fPower, gamma_fPower); % α/γ -> Lower alpha and higher gamma is common pre-seizure


    b_g = safe(beta_fPower, gamma_fPower); % β/γ -> Fast rhythms are affected during pre-ictal and ictal periods

    b_at = safe(beta_fPower, alpha_fPower + theta_fPower); % β/(α+θ) -> Ratio reflecting activation states and seizure network involvement
    t_ab = safe(theta_fPower, alpha_fPower + beta_fPower); % θ/(α+β) -> Drowsiness and slowing tendency increases pre-ictally
    dt_ab = safe(delta_fPower + theta_fPower, alpha_fPower + beta_fPower); % (δ+θ)/(α+β) -> slowing vs. activation (activation = capturing epileptiforms -> IDing seizures)

    % high frequency power drops in the pre-ictal period and low frequency power increases
    low_high  = safe(delta_fPower + theta_fPower, beta_fPower + gamma_fPower); % Low/High: should increase in the pre-ictal period
    high_low  = safe(beta_fPower + gamma_fPower, delta_fPower + theta_fPower); % High/Low: should decrease in the pre-ictal period
end

    % Spectral Features
function [spectralEntropy, PSD, PSD_norm, FT, FT_norm, totalPower] = feature_domain_spectral_features(signal, Fs)

    signal = double(signal);
    [numCh, nSamples] = size(signal);

    % Spectral entropy
    spectralEntropy = zeros(numCh, 1);

    % FFT and PSD settings
    nfft = max(2^nextpow2(nSamples), 1024);
    PSD  = zeros(numCh, nfft/2+1);
    PSD_norm = zeros(numCh, nfft/2+1);

    FT = zeros(numCh, nSamples);
    FT_norm = zeros(numCh, nSamples);

    % Total power for each channel
    totalPower = zeros(numCh, 1);

    % Welch parameters (2-second window or full length)
    winLength = min(round(Fs*2), nSamples);
    overlap   = floor(winLength/2);

    for ch = 1:numCh
        x = signal(ch, :);

        % 1. Spectral Entropy
        spectralEntropy(ch) = spectralEntropy(x, Fs);

        % 2. PSD 
        [Pxx, ~] = pwelch(x, hamming(winLength), overlap, nfft, Fs);
        PSD(ch,:) = Pxx;

        % Normalized PSD
        PSD_norm(ch,:) = Pxx ./ sum(Pxx);

        % 3. FFT Power Spectrum
        X = abs(fft(x)).^2;
        FT(ch,:) = X;

        % Normalized FFT
        FT_norm(ch,:) = X ./ sum(X);

        % 4. Total Power
        df = Fs / nfft;
        totalPower(ch) = sum(Pxx)*df; %∫PSD df
    end
end

%% Extracting the features
Fs = 256; % Sampled at 256 samples per second
inputDir = 'Filtered_Data'; % I CHANGED THIS FROM data_truncated TO Filtered_Data LMK IF YOU HAD IT THAT WAY FOR A REASON!
outputDir = "Dataset";

files = dir(fullfile(inputDir, '**', '*.mat'));
fprintf("Found %d files.\n", numel(files));

%% Start filing
for i = 1:numel(files)
    fprintf("\nProcessing %d of %d: %s\n", i, numel(files), files(i).name);
    % 1. Extracting the EEG
    filePath = fullfile(files(i).folder, files(i).name);
    data = load(filePath);
    
    if ~isfield(data, 'segment')
        warning("File %s has no 'segment' field. Skipping.\n", files(i).name);
        continue;
    end

    eeg = data.segment; 
    
    % 2. Extracting features
    [meanEEG, stdEEG, kurtEEG, skewEEG, AR_coeff] = time_domain_features(eeg);
    [activityEEG, mobilityEEG, complexityEEG] = hjorth_parameters(eeg);
    [alpha_fPower, beta_fPower, gamma_fPower, delta_fPower, theta_fPower] = feature_domain_bandPower(eeg, Fs);
    [d_a, d_b, d_g, d_t, t_a, t_b, t_g, a_b, a_g, b_g, b_at, t_ab, dt_ab, low_high, high_low] = compute_bandpower_ratios(alpha_fPower, beta_fPower, gamma_fPower, delta_fPower, theta_fPower);
    [spectral_Entropy, PSD, FT] = feature_domain_spectral_features(eeg, Fs);

    % 3. Saving the extracted features
    % Convert AR_coeff cell array (vectors) to cell array of strings
AR_str = cellfun(@(v) strjoin(string(v), ';'), AR_coeff, 'UniformOutput', false);

% Then create the table including AR_str as one column
features = table(meanEEG, stdEEG, kurtEEG, skewEEG, AR_str, ...
                 activityEEG, mobilityEEG, complexityEEG, ...
                 alpha_fPower, beta_fPower, gamma_fPower, delta_fPower, theta_fPower, ...
                 d_a, d_b, d_g, d_t, t_a, t_b, t_g, a_b, a_g, b_g, b_at, t_ab, dt_ab, low_high, high_low, ...
                 spectral_Entropy, ...
                 'VariableNames', {'MeanEEG', 'StdEEG', 'Kurtosis', 'Skewness', 'AR_Coefficients', ...
                                   'HjorthActivity', 'HjorthMobility', 'HjorthComplexity', ...
                                   'AlphaPower', 'BetaPower', 'GammaPower', 'DeltaPower', 'ThetaPower', ...
                                   'delta/alpha', 'delta/beta', 'delta/gamma', 'delta/theta', 'theta/alpha', 'theta/beta', 'theta/gamma', 'alpha/beta', 'alpha/gamma', 'beta/gamma', 'beta/(alpha+theta)', 'theta/(alpha+beta)', '(delta+theta)/(alpha+beta)', '(delta+theta)/(beta+gamma)', '(beta+gamma)/(delta+theta)', ...
                                   'SpectralEntropy'});
    writetable(features, outFile)

    % 4. Saving the features
        % Creating the New Subfolders
    relPath = files(i).folder(length(inputDir)+1:end);
    if startsWith(relPath, filesep)
        relPath = relPath(2:end);
    end

        % Path for the New Subfolder
    outFolder = fullfile(outputDir, relPath);

        % Create a New Folder if it Does Not Exist
    if ~exist(outFolder, 'dir')
        mkdir(outFolder);
    end
        % Names for the New Filtered Data
    [~, base, ~] = fileparts(files(i).name);
    outFile = fullfile(outFolder, base + ".csv"); 

    save(outFile, "bp_eeg_"); % Append the transposed data
end

