%% Part 2: Feature Extraction using EWT (Empirical Wavelet Transform) - 70/30 Split
clc; clear; close all;

disp('------------------------------------------------------');
disp('POWER QUALITY DISTURBANCE CLASSIFICATION USING EWT');
disp('Empirical Wavelet Transform-based Feature Extraction');
disp('70/30 Train/Test Split');
disp('------------------------------------------------------');

% Load dataset
load('PQD_Signals.mat');
disp(['Loaded dataset with ', num2str(size(z, 1)), ' signals']);

%% Dataset Splitting (70% train, 30% test)
rng(42); % For reproducibility
train_indices = [];
test_indices = [];

% Create indices for each class
unique_classes = unique(cl);
disp(['Number of classes: ', num2str(length(unique_classes))]);

% Perform stratified sampling for each class
for c = 1:length(unique_classes)
    class_name = unique_classes{c};
    indices = find(strcmp(cl, class_name));
    
    % Shuffle indices
    indices = indices(randperm(length(indices)));
    
    % Calculate split points (70% train, 30% test)
    train_count = round(length(indices) * 0.7);
    
    % Split indices
    train_indices = [train_indices; indices(1:train_count)];
    test_indices = [test_indices; indices(train_count+1:end)];
end

% Create datasets
train_data = z(train_indices, :);
train_labels = cl(train_indices);
test_data = z(test_indices, :);
test_labels = cl(test_indices);

disp(['Training set: ', num2str(size(train_data, 1)), ' signals']);
disp(['Test set: ', num2str(size(test_data, 1)), ' signals']);

%% Parameters for EWT-based Feature Extraction
num_IMFs = 5;                   % Number of EWT modes to extract
features_per_IMF = 12;          % Features per EWT mode
global_features = 5;            % Global signal features
total_features = num_IMFs * features_per_IMF + global_features;

fprintf('\nBeginning EWT-based feature extraction process...\n');
fprintf('Using EWT to decompose signals into %d modes\n', num_IMFs);

%% Initialize Feature Matrices
train_features = zeros(size(train_data, 1), total_features);
test_features = zeros(size(test_data, 1), total_features);

%% Process All Signals with EWT
fprintf('Extracting EWT-based features from training signals...\n');
for i = 1:size(train_data, 1)
    train_features(i,:) = extract_EWT_features(train_data(i,:), fs, num_IMFs);
    if mod(i, 50) == 0 || i == size(train_data, 1)
        fprintf('Processed %d/%d training signals with EWT\n', i, size(train_data, 1));
    end
end

fprintf('Extracting EWT-based features from test signals...\n');
for i = 1:size(test_data, 1)
    test_features(i,:) = extract_EWT_features(test_data(i,:), fs, num_IMFs);
    if mod(i, 20) == 0 || i == size(test_data, 1)
        fprintf('Processed %d/%d test signals with EWT\n', i, size(test_data, 1));
    end
end

%% Feature Normalization
all_features = [train_features; test_features];
min_vals = min(all_features);
max_vals = max(all_features);
range_vals = max_vals - min_vals;
range_vals(range_vals == 0) = 1;

train_features_norm = (train_features - min_vals) ./ range_vals;
test_features_norm = (test_features - min_vals) ./ range_vals;

% Replace any NaN or Inf values
train_features_norm(isnan(train_features_norm) | isinf(train_features_norm)) = 0;
test_features_norm(isnan(test_features_norm) | isinf(test_features_norm)) = 0;

%% Save EWT-based Features
save('PQD_features_EWT_70_30.mat', ...
     'train_features_norm', 'test_features_norm', ...
     'train_labels', 'test_labels', 'unique_classes');

fprintf('\nEWT-based feature extraction completed. Saved to PQD_features_EWT_70_30.mat\n');
fprintf('Total features extracted: %d features per signal\n', total_features);
fprintf('   - %d features per EWT mode × %d modes\n', features_per_IMF, num_IMFs);
fprintf('   - %d global signal features\n', global_features);

%% EWT Feature Extraction Function
function features = extract_EWT_features(signal, fs, num_IMFs)
    % Parameters
    features_per_IMF = 12;
    global_features = 5;
    total_features = num_IMFs * features_per_IMF + global_features;
    f0 = 50;
    
    % Initialize feature vector
    features = zeros(1, total_features);
    
    % 1. APPLY EMPIRICAL WAVELET TRANSFORM (EWT)
    % ------------------------------------------
    ewt_success = false;
    
    try
        % CONFIGURATION 1: Standard EWT with localized maxima method
        params = struct();
        params.SamplingRate = fs;
        params.log = 0;
        params.method = 'locmax';
        params.detect = 'locmax';
        params.completion = 0;
        params.InitBounds = 0;
        params.MaxNumPeaks = num_IMFs;
        params.typeDetect = 'otsu';
        
        % Apply EWT - main implementation
        [ewt_coeffs, mra, boundaries] = ewt(signal, params);
        
        % Check if EWT succeeded
        if ~isempty(mra) && size(mra, 2) >= num_IMFs
            ewt_success = true;
        end
    catch
        ewt_success = false;
    end
    
    % Try alternate EWT configuration if first one failed
    if ~ewt_success
        try
            % CONFIGURATION 2: Scale-space approach
            params = struct();
            params.SamplingRate = fs;
            params.log = 0;
            params.method = 'scalespace';
            params.detect = 'scalespace';
            params.reg = 'average';
            params.numemuband = num_IMFs;
            
            % Apply EWT with scale-space configuration
            [ewt_coeffs, mra, boundaries] = ewt(signal, params);
            
            % Check if EWT succeeded
            if ~isempty(mra) && size(mra, 2) >= num_IMFs
                ewt_success = true;
            end
        catch
            ewt_success = false;
        end
    end
    
    % If both EWT attempts failed, use filter bank decomposition as fallback
    if ~ewt_success || size(mra, 2) < num_IMFs
        % Use filter bank as fallback
        mra = zeros(length(signal), num_IMFs);
        
        % Filter bank frequency bands
        cutoff_freqs = [0 50 100 200 500 fs/2];
        
        % Design and apply filters for each band
        for i = 1:num_IMFs
            try
                if i == 1 % Highest band
                    [b, a] = butter(4, [cutoff_freqs(end-1)/(fs/2), 0.95], 'bandpass');
                elseif i == num_IMFs % Lowest band
                    [b, a] = butter(4, cutoff_freqs(2)/(fs/2), 'low');
                else % Middle bands
                    [b, a] = butter(4, [cutoff_freqs(num_IMFs-i+1)/(fs/2), cutoff_freqs(num_IMFs-i+2)/(fs/2)], 'bandpass');
                end
                mra(:, i) = filtfilt(b, a, signal);
            catch
                % If filtering fails, use simple band-limited signal
                mra(:, i) = signal .* (1/i);
            end
        end
    end
    
    % Ensure we have exactly num_IMFs components
    if size(mra, 2) > num_IMFs
        mra = mra(:, 1:num_IMFs); % Take first num_IMFs modes
    elseif size(mra, 2) < num_IMFs
        % Pad with zeros if needed
        mra = [mra, zeros(length(signal), num_IMFs - size(mra, 2))];
    end
    
    % 2. EXTRACT FEATURES FROM EWT MODES
    % ----------------------------------
    for imf_idx = 1:num_IMFs
        imf = mra(:, imf_idx);
        start_idx = (imf_idx-1)*features_per_IMF + 1;
        
        % Time-domain features (7)
        features(start_idx) = sqrt(mean(imf.^2));                % RMS
        features(start_idx+1) = max(imf) - min(imf);             % Peak-to-peak
        features(start_idx+2) = mean(abs(imf));                  % Mean absolute value
        features(start_idx+3) = std(imf + eps);                  % Standard deviation
        
        % Skip skewness/kurtosis if signal is too uniform
        if all(imf == 0) || length(unique(imf)) <= 1
            features(start_idx+4) = 0; % Skewness
            features(start_idx+5) = 0; % Kurtosis
        else
            features(start_idx+4) = skewness(imf);              % Skewness
            features(start_idx+5) = kurtosis(imf);              % Kurtosis  
        end
        
        % Zero crossing rate
        features(start_idx+6) = sum(abs(diff(sign(imf))))/2/length(imf);
        
        % Frequency domain features - safe FFT-based approach
        N = length(imf);
        X = abs(fft(imf))/N;
        X = X(1:floor(N/2)+1);
        X(2:end-1) = 2*X(2:end-1);
        f = (0:floor(N/2))*fs/N;
        
        % Find dominant frequency
        [~, max_idx] = max(X);
        if max_idx <= length(f)
            features(start_idx+7) = f(max_idx);                 % Dominant frequency
        else
            features(start_idx+7) = 0;
        end
        
        % Spectral band energies
        total_energy = sum(X.^2) + eps;
        
        % Low frequency band energy (0-50Hz)
        low_idx = f <= f0;
        if any(low_idx)
            features(start_idx+8) = sum(X(low_idx).^2)/total_energy;
        else
            features(start_idx+8) = 0;
        end
        
        % High frequency band energy (>50Hz)
        high_idx = f > f0;
        if any(high_idx)
            features(start_idx+9) = sum(X(high_idx).^2)/total_energy;
        else
            features(start_idx+9) = 0;
        end
        
        % Instantaneous features from Hilbert transform
        try
            analytic_signal = hilbert(imf);
            inst_amplitude = abs(analytic_signal);
            inst_phase = angle(analytic_signal);
            
            % Instantaneous amplitude statistical feature
            features(start_idx+10) = mean(inst_amplitude);       % Mean amplitude
            
            % Instantaneous frequency approximation
            inst_freq = [0; diff(unwrap(inst_phase))] * fs / (2*pi);
            inst_freq(inst_freq < 0) = 0;
            features(start_idx+11) = mean(inst_freq);            % Mean frequency
        catch
            features(start_idx+10:start_idx+11) = [0, 0];
        end
    end
    
    % 3. GLOBAL FEATURES
    % -----------------
    % Entropy (approximation using histogram)
    try
        p = histcounts(signal, min(50, length(signal)/10), 'Normalization', 'probability');
        p = p(p > 0);
        features(end-4) = -sum(p .* log2(p));                   % Shannon entropy
    catch
        features(end-4) = 0;
    end
    
    % Number of peaks
    try
        [~, locs] = findpeaks(signal, 'MinPeakProminence', 0.1*std(signal));
        features(end-3) = length(locs);                         
    catch
        features(end-3) = 0;
    end
    
    % Total harmonic distortion (THD)
    try
        % Find fundamental frequency component
        N = length(signal);
        X = abs(fft(signal))/N;
        X = X(1:floor(N/2)+1);
        f = (0:floor(N/2))*fs/N;
        
        % Find fundamental around 50Hz
        f_range = (f >= 45) & (f <= 55);
        if any(f_range)
            [~, max_idx] = max(X(f_range));
            fund_idx = find(f_range, 1) + max_idx - 1;
            
            if fund_idx > 0 && fund_idx*5 <= length(X)
                % Calculate harmonics power to fundamental power ratio
                harm_indices = [2*fund_idx, 3*fund_idx, 4*fund_idx, 5*fund_idx];
                harm_indices = harm_indices(harm_indices <= length(X));
                
                if ~isempty(harm_indices)
                    harm_power = sum(X(harm_indices).^2);
                    fund_power = X(fund_idx)^2;
                    if fund_power > 0
                        features(end-2) = sqrt(harm_power/fund_power);  % THD
                    else
                        features(end-2) = 0;
                    end
                else
                    features(end-2) = 0;
                end
            else
                features(end-2) = 0;
            end
        else
            features(end-2) = 0;
        end
    catch
        features(end-2) = 0;
    end
    
    % Signal-to-noise ratio (SNR) approximations
    try
        % Use median filtering to separate signal from noise
        signal_approx = medfilt1(signal, 5);
        noise_approx = signal - signal_approx;
        
        % Calculate SNR and SINAD
        signal_power = sum(signal_approx.^2);
        noise_power = sum(noise_approx.^2) + eps;
        
        features(end-1) = 10*log10(signal_power/noise_power);    % SNR estimate
        features(end) = 10*log10(sum(signal.^2)/(sum((signal-mean(signal)).^2) + eps));  % SINAD estimate
    catch
        features(end-1:end) = [40, 40];
    end
    
    % Final check for NaN or Inf values
    features(isnan(features) | isinf(features)) = 0;
end