%% Power Quality Disturbance Signal Generator (9 Types)
% Exactly matches the mathematical models from your paper
fs = 3200;                  % Sampling frequency (Hz)
T = 1/50;                   % Fundamental period (s)
ts = 1/fs;                  % Sampling interval
t = 0:ts:10*T-ts;           % 10 cycles duration

% Common parameters
f0 = 49.5 + rand()*1.0;     % 49.5 ≤ f ≤ 50.5 Hz (paper compliant)
phi = rand()*pi;            % 0 ≤ φ ≤ 180° (paper compliant)
num_signals_per_type = 100; % 100 signals per type
SNR = 40;                   % Signal-to-noise ratio (dB)

% Initialize storage (9 classes: C2-C10)
z = []; cl = {}; 
class_names = {'Sag', 'Sag+Harmonics', 'Swell', 'Swell+Harmonics', ...
               'Harmonics', 'Flicker', 'Oscillatory Transient', 'Notch', 'Spike'};

%% Generate signals for each class
for class_idx = 1:9
    for sig_num = 1:num_signals_per_type
        % Common random parameters
        t1 = 2*T + 3*T*rand();      % Start time (2T-5T)
        duration = T + 8*T*rand();   % Base duration (T-9T)
        t2 = t1 + duration;
        
        switch class_idx
            case 1 % C2 - Sag (paper exact)
                a = 0.1 + 0.8*rand(); % 0.1 ≤ a ≤ 0.9
                sag = (1 - a*(heaviside(t-t1) - heaviside(t-t2)));
                y = sag .* sin(2*pi*f0*t - phi);
                
            case 2 % C3 - Sag + Harmonics (paper exact)
                a = 0.1 + 0.8*rand(); % 0.1 ≤ a ≤ 0.9
                sag = (1 - a*(heaviside(t-t1) - heaviside(t-t2)));
                fundamental = sag .* sin(2*pi*f0*t - phi);
                
                % Random harmonics (0.03 ≤ a_i ≤ 0.25)
                harmonics = (0.03 + 0.22*rand())*sin(3*2*pi*f0*t - phi) + ...
                           (0.03 + 0.22*rand())*sin(5*2*pi*f0*t - phi) + ...
                           (0.03 + 0.22*rand())*sin(7*2*pi*f0*t - phi);
                y = fundamental + harmonics;
                
            case 3 % C4 - Swell (paper exact)
                a = 0.1 + 0.7*rand(); % 0.1 ≤ a ≤ 0.8
                swell = (1 + a*(heaviside(t-t1) - heaviside(t-t2)));
                y = swell .* sin(2*pi*f0*t - phi);
                
            case 4 % C5 - Swell + Harmonics (paper exact)
                a = 0.1 + 0.7*rand(); % 0.1 ≤ a ≤ 0.8
                swell = (1 + a*(heaviside(t-t1) - heaviside(t-t2)));
                fundamental = swell .* sin(2*pi*f0*t - phi);
                
                % Random harmonics (0.03 ≤ a_i ≤ 0.25)
                harmonics = (0.03 + 0.22*rand())*sin(3*2*pi*f0*t - phi) + ...
                           (0.03 + 0.22*rand())*sin(5*2*pi*f0*t - phi) + ...
                           (0.03 + 0.22*rand())*sin(7*2*pi*f0*t - phi);
                y = fundamental + harmonics;
                
            case 5 % C6 - Harmonics (paper exact)
                fundamental = sin(2*pi*f0*t - phi);
                
                % Random harmonics (0.03 ≤ a_i ≤ 0.25)
                harmonics = (0.03 + 0.22*rand())*sin(3*2*pi*f0*t - phi) + ...
                           (0.03 + 0.22*rand())*sin(5*2*pi*f0*t - phi) + ...
                           (0.03 + 0.22*rand())*sin(7*2*pi*f0*t - phi) + ...
                           (0.03 + 0.22*rand())*sin(9*2*pi*f0*t - phi) + ...
                           (0.03 + 0.22*rand())*sin(11*2*pi*f0*t - phi);
                y = fundamental + harmonics;
                
            case 6 % C7 - Flicker (paper exact)
                a = 0.1 + 0.1*rand(); % 0.1 ≤ a ≤ 0.2
                B = 5 + 20*rand();    % 5 ≤ B ≤ 25
                flicker = (1 + a*sin(2*pi*B*t));
                y = flicker .* sin(2*pi*f0*t - phi);
                
            case 7 % C8 - Oscillatory Transient (paper exact)
                a = 0.1 + 0.7*rand(); % 0.1 ≤ a ≤ 0.8
                tau = 0.008 + 0.032*rand(); % 8ms ≤ τ ≤ 40ms
                ft = 300 + 3200*rand(); % 300 ≤ f_t ≤ 3500
                duration = 0.5*T + 2.5*T*rand(); % 0.5T ≤ duration ≤ 3T
                t2 = t1 + duration;
                
                transient = a*exp(-(t-t1)/tau).*(heaviside(t-t1)-heaviside(t-t2))...
                           .*sin(2*pi*ft*t);
                y = sin(2*pi*f0*t - phi) + transient;
                
            case 8 % C9 - Notch (paper exact)
                a = 0.1 + 0.3*rand(); % 0.1 ≤ a ≤ 0.4
                notch_width = (0.01 + 0.04*rand())*T; % 0.01T ≤ width ≤ 0.05T
                n = randi([0 5]); % Multiple notches (0 ≤ n ≤ 5)
                t2 = t1 + notch_width;
                
                y = sin(2*pi*f0*t - phi) - ...
                    a*sign(sin(2*pi*f0*t - phi)).*...
                    (heaviside(t-(t1+0.02*n*T)) - heaviside(t-(t2+0.02*n*T)));
                
            case 9 % C10 - Spike (paper exact)
                a = 0.1 + 0.3*rand(); % 0.1 ≤ a ≤ 0.4
                spike_width = (0.01 + 0.04*rand())*T; % 0.01T ≤ width ≤ 0.05T
                n = randi([0 5]); % Multiple spikes (0 ≤ n ≤ 5)
                t2 = t1 + spike_width;
                
                y = sin(2*pi*f0*t - phi) + ...
                    a*sign(sin(2*pi*f0*t - phi)).*...
                    (heaviside(t-(t1+0.02*n*T)) - heaviside(t-(t2+0.02*n*T)));
        end
        
        % Add noise and store
        y = awgn(y, SNR);
        z = [z; y];
        cl = [cl; class_names{class_idx}];
    end
end

%% Dataset splitting (900 signals total - 100 per 9 classes)
numSignalsTotal = size(z, 1);
[trainInd, valInd, testInd] = dividerand(numSignalsTotal, 0.7, 0.2, 0.1);

train_data = z(trainInd, :);
train_labels = cl(trainInd);
val_data = z(valInd, :);
val_labels = cl(valInd);
test_data = z(testInd, :);
test_labels = cl(testInd);

%% Save datasets
save('PQD_train_data.mat', 'train_data', 'train_labels', '-v7.3');
save('PQD_val_data.mat', 'val_data', 'val_labels', '-v7.3');
save('PQD_test_data.mat', 'test_data', 'test_labels', '-v7.3');

disp(['Signal generation complete. Created ', num2str(numSignalsTotal), ' signals:']);
disp(['- Training: ', num2str(length(trainInd)), ' signals']);
disp(['- Validation: ', num2str(length(valInd)), ' signals']);
disp(['- Testing: ', num2str(length(testInd)), ' signals']);