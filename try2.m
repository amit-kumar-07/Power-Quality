%% Part 2: Feature Extraction using EWT and HT
clc; clear; close all;

% Load datasets
load('PQD_train_data.mat');
load('PQD_val_data.mat');
load('PQD_test_data.mat');

%% Parameters
num_IMFs = 5;                   % Number of IMFs to extract
fs = 3200;                      % Sampling frequency (Hz)
f0 = 50;                        % Fundamental frequency (Hz)
features_per_IMF = 14;          % Fixed number of features per IMF
global_features = 5;            % Global signal features
total_features = num_IMFs * features_per_IMF + global_features;

%% Initialize Feature Matrices
train_features = zeros(size(train_data, 1), total_features);
val_features = zeros(size(val_data, 1), total_features);
test_features = zeros(size(test_data, 1), total_features);

%% Standard Feature Names
feature_names = {};
for imf_idx = 1:num_IMFs
    prefix = sprintf('IMF%d_', imf_idx);
    feature_names = [feature_names, ...
                    strcat(prefix, 'RMS'), ...
                    strcat(prefix, 'Peak2Peak'), ...
                    strcat(prefix, 'MeanAbs'), ...
                    strcat(prefix, 'Std'), ...
                    strcat(prefix, 'Skewness'), ...
                    strcat(prefix, 'Kurtosis'), ...
                    strcat(prefix, 'DomFreq'), ...
                    strcat(prefix, 'HF_Energy'), ...
                    strcat(prefix, 'LF_Energy'), ...
                    strcat(prefix, 'HT_MeanAmp'), ...
                    strcat(prefix, 'HT_StdAmp'), ...
                    strcat(prefix, 'HT_MeanFreq'), ...
                    strcat(prefix, 'HT_StdFreq')];
end
feature_names = [feature_names, ...
                'Signal_Entropy', ...
                'Num_Peaks', ...
                'THD', ...
                'SINAD', ...
                'SNR'];

%% Feature Extraction Function
extract_features = @(signal) ...
    extract_PQD_features(signal, fs, f0, num_IMFs, features_per_IMF, global_features);

%% Process All Signals
fprintf('Extracting training features...\n');
for i = 1:size(train_data, 1)
    train_features(i,:) = extract_features(train_data(i,:));
end

fprintf('Extracting validation features...\n');
for i = 1:size(val_data, 1)
    val_features(i,:) = extract_features(val_data(i,:));
end

fprintf('Extracting test features...\n');
for i = 1:size(test_data, 1)
    test_features(i,:) = extract_features(test_data(i,:));
end

%% Feature Normalization
all_features = [train_features; val_features; test_features];
min_vals = min(all_features);
max_vals = max(all_features);

train_features_norm = (train_features - min_vals) ./ (max_vals - min_vals);
val_features_norm = (val_features - min_vals) ./ (max_vals - min_vals);
test_features_norm = (test_features - min_vals) ./ (max_vals - min_vals);

%% Save Results
save('PQD_features_final.mat', ...
     'train_features_norm', 'val_features_norm', 'test_features_norm', ...
     'feature_names', 'train_labels', 'val_labels', 'test_labels');

fprintf('Feature extraction completed. Saved to PQD_features_final.mat\n');

%% Helper Function
function features = extract_PQD_features(signal, fs, f0, num_IMFs, features_per_IMF, global_features)
    features = zeros(1, num_IMFs * features_per_IMF + global_features);
    
    % 1. EWT Decomposition
    try
        [mra, ~] = ewt(signal, 'MaxNumPeaks', num_IMFs);
    catch
        mra = zeros(length(signal), num_IMFs);
    end
    
    % 2. Process each IMF
    for imf_idx = 1:min(size(mra, 2), num_IMFs)
        imf = mra(:, imf_idx);
        start_idx = (imf_idx-1)*features_per_IMF + 1;
        
        % Time-domain features (6)
        features(start_idx:start_idx+5) = [...
            rms(imf), peak2peak(imf), mean(abs(imf)), ...
            std(imf), skewness(imf), kurtosis(imf)];
        
        % Frequency-domain features (3)
        [psd, freq] = pwelch(imf, [], [], [], fs);
        [~, idx] = max(psd);
        features(start_idx+6:start_idx+8) = [...
            freq(idx), sum(psd(freq > f0*2)), sum(psd(freq <= f0*2))];
        
        % Hilbert Transform features (4)
        analytic_signal = hilbert(imf);
        inst_amplitude = abs(analytic_signal);
        inst_freq = diff(unwrap(angle(analytic_signal))) * fs / (2*pi);
        features(start_idx+9:start_idx+12) = [...
            mean(inst_amplitude), std(inst_amplitude), ...
            mean(inst_freq), std(inst_freq)];
    end
    
    % 3. Global features (5)
    features(end-4:end) = [...
        entropy(signal), ...
        length(findpeaks(signal)), ...
        thd(signal, fs), ...
        sinad(signal, fs), ...
        snr(signal, fs)];
end