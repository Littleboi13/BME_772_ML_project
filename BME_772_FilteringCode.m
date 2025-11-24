%% BME 772 Final Project Filtering Code
% Data Truncation -> *Filtering* -> Feature Extraction -> Machine Learning
% Filtering: Detrend -> Notch (Fundamental + Harmonic) -> Bandpass
% Student IDs: Jacklyn D'Ascenzo (xxxxx7753), Mya Hao (xxxxx8133), Tanvir Hassan (xxxxx4056), Janneza Macaalay (xxxxx2196)

%% Detrending and Removal of DC Offset
function clean_EEG = cleaning(signal)
    % Removing the NaN channels in the Data
    signal(all(isnan(signal), 2), :) = [];

    % Detrending the data
    signal = detrend(signal')';

    % Removal of DC Offset
    clean_EEG = signal - mean(signal, 2); % remove DC offset per channel
end

%% Notch Filter Function
    % Finding the Powerline Noise -> Either 50Hz or 60Hz depending on the equipment
        % Hospitals Use 60Hz Power Sources (North American Devices) and 50Hz Power Sources (Internationally Imported Devices)
function [lineFreqFund, lineFreqHarm] = detect_line_frequency(signal, Fs)
    % Only using one EEG channel as reference for this
    ref = signal(1,:);

    % Finding PSD with pwelch
    desiredWin = round(Fs*2);
    winLength = min(desiredWin, length(ref));   
    win = hamming(winLength);
    overlap = floor(winLength*0.5); 
    nfft = max(2^nextpow2(winLength), 1024); %(Δf = Fs/NFFT)
    
    [pxx, f] = pwelch(ref, win, overlap, nfft, Fs); % FFT of each to get Power -> hamming is a windowing function

    % Finding the Powerline Noise Frequency Fundamental Frequency
    idx_fund = f >= 45 & f <= 65;
    [~, i1] = max(pxx(idx_fund));
    lineFreqFund = f(idx_fund);
    lineFreqFund = lineFreqFund(i1);

    % Finding the Harmonics of the Powerline Noise
    idx_harm = f >= 95 & f <= 125;
    [~, i2] = max(pxx(idx_harm));
    lineFreqHarm = f(idx_harm);
    lineFreqHarm = lineFreqHarm(i2);
end

    % Notch Filter for Determined Powerline Noise
function sig_filtered = apply_notch(signal, Fs, lineFreqFund, lineFreqHarm)
    Q = 60;  % notch sharpness (higher = narrower band)
    sig_filtered = zeros(size(signal)); % initializing the output variable

    for ch = 1:size(signal, 1)
        sig = double(signal(ch,:)); % ensuring type double for iirnotch function

        % Fundamental Frequency Notch
        if lineFreqFund > 0 && lineFreqFund < Fs/2 % freq cannot be negative and it must be less than half the sampling
            W0 = lineFreqFund/(Fs/2);
            BW = W0/Q;
            [bF, aF] = iirnotch(W0, BW);
            sig = filtfilt(bF, aF, sig);
        end

        % Harmonic Frequency Notch
        if lineFreqHarm > 0 && lineFreqHarm < Fs/2
            W0 = lineFreqHarm/(Fs/2);
            BW = W0/Q;
            [bH, aH] = iirnotch(W0, BW);
            sig = filtfilt(bH, aH, sig);
        end

        sig_filtered(ch, :) = sig;
    end
end

%% Bandpass Filtering
    % Broad Bandpass for Removing Baseline Drift and High Frequency EMG Noise
function sig_bp = broad_bandpass(signal, Fs)
    Fpass = [0.5 60]; % 0.5 (min of delta) – 60 (max of gamma) Hz bandpass for EEG
    order = 4; % 4th order Butterworth
    [bB,aB] = butter(order, Fpass/(Fs/2), 'bandpass');
    sig_bp = zeros(size(signal)); % initializing the bandpass output vector

    for ch = 1:size(signal, 1)
        sig_bp(ch,:)  = filtfilt(bB, aB, double(signal(ch, :)));
    end
end

%% Applying the Filtering Sequence
Fs = 256; % Sampled at 256 samples per second
inputDir = 'data_truncated';
outputDir = "Filtered_Data";

files = dir(fullfile(inputDir, '**', '*.mat'));
fprintf("Found %d files.\n", numel(files));

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

    %2. Detrending and DC Offset Removal
    clean_eeg = cleaning(eeg);

    % 3. Finding the Powerline Noise
    [lineFund, lineHarm] = detect_line_frequency(clean_eeg, Fs);

    % 4. Notch Filter for Powerline Noise
    notch_eeg = apply_notch(clean_eeg, Fs, lineFund, lineHarm);

    % 5. Broad Bandpass
    bp_eeg = broad_bandpass(notch_eeg, Fs);

    % 7. Saving the Clean Signals
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
    outFile = fullfile(outFolder, base + "_filtered.mat"); 

     % Transpose the matrix before saving
    save(outFile, "bp_eeg"); % Append the transposed data

    % % Plotting only the first EEG signal
    % if i == 1
    %     fprintf("Plotting first file: %s\n", files(i).name);
    % 
    %     t = (0:size(eeg,2)-1) / Fs;   % time axis
    % 
    %     figure;
    %     subplot(2,1,1);
    %     plot(t, eeg(1,:), "Color", "#9267FF");   % Channel 1
    %     title('Raw Input EEG Data (Channel 1)');
    %     xlabel('Time (s)');
    %     ylabel('Amplitude');
    %     ylim([-600 600])
    %     grid on;
    % 
    %     subplot(2,1,2);
    %     plot(t, bp_eeg(1,:), "Color", "#9267FF");
    %     title('Bandpass Filtered EEG Data (Channel 1)');
    %     xlabel('Time (s)');
    %     ylabel('Amplitude');
    %     ylim([-600 600])
    %     grid on;
    % 
    %     sgtitle(sprintf('EEG Processing Stages for: %s', files(i).name), 'Interpreter', 'none');
    % 
    %     % Frequency Plots
    %     figure;
    %     subplot(2,1,1);
    %     [pxx_raw, f_raw] = pwelch(eeg(1,:), hamming(1024), 512, 1024, Fs);
    %     plot(f_raw, 10*log10(pxx_raw), "Color", "#9267FF", "LineWidth", 1.3);
    %     grid on;
    %     xlabel('Frequency (Hz)');
    %     ylabel('PSD (dB/Hz)');
    %     title('Raw Input EEG PSD (Channel 1)');
    % 
    %     subplot(2,1,2);
    %     [pxx_bp, f_bp] = pwelch(bp_eeg(1,:), hamming(1024), 512, 1024, Fs);
    %     plot(f_bp, 10*log10(pxx_bp), "Color", "#9267FF", "LineWidth", 1.3);
    %     grid on;
    %     xlabel('Frequency (Hz)');
    %     ylabel('PSD (dB/Hz)');
    %     title('Bandpass Filtered EEG PSD (Channel 1)');
    % 
    %     sgtitle(sprintf('EEG PSD Stages for: %s', files(i).name), 'Interpreter', 'none');
    % end
end