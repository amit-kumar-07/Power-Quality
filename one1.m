clc;
clear;
close all;

% Parameters
fs = 3200; % Sampling frequency (Hz)
T = 1/50; % Fundamental period (s)
ts = 1/fs; % Sampling interval
t = 0:ts:0.2-ts; % 200ms duration (as per IEC 61000-4-7 standard)
f0 = 49.5 + rand()*1.0; % Random fundamental frequency (49.5 ≤ f ≤ 50.5 Hz)
phi = rand()*pi; % Random phase angle (0 ≤ φ ≤ π)

% Initialize storage
num_signals_per_type = 100; % Number of signals per type
SNR = 40; % Signal-to-noise ratio (dB)
z = [];
cl = {};

% Disturbance types
class_names = {'Normal', 'Swell+Transient', 'Sag', ...
               'Swell', 'Notch+Transient', 'Interrupt', 'Flicker+Sag', ...
               'Harmonics', 'Transient', 'Flicker', 'Notch', 'Spike', ...
               'Spike+Harmonics', 'Sag+Harmonics', 'Swell+Harmonics', ...
               'Interrupt+Harmonics', 'Transient+Harmonics', ...
               'Flicker+Harmonics', 'Sag+Transient'};

% Generate signals
for class_idx = 1:length(class_names)
    fprintf('Generating %s signals (%d/%d)...\n', class_names{class_idx}, class_idx, length(class_names));
    
    for sig_num = 1:num_signals_per_type
        % Randomize frequency and phase slightly for each signal
        f0 = 49.5 + rand()*1.0;
        phi = rand()*pi;
        
        switch class_idx
            case 1 % Normal
                y = sin(2*pi*f0*t - phi);
                
            case 2 % Swell + Transient
                swell_amplitude = 1.2 + 0.4*rand(); % 1.2-1.6 range for variety
                swell = swell_amplitude * sin(2*pi*f0*t - phi);
                transient_start = 0.05 + 0.03*rand(); % Random start time
                transient_freq = 250 + 100*rand(); % Random frequency
                transient = 0.8 * exp(-200*(t-transient_start).^2) .* sin(2*pi*transient_freq*t);
                y = swell + transient;

            case 3 % Sag
                sag_amplitude = 0.4 + 0.2*rand(); % 0.4-0.6 range
                y = sag_amplitude * sin(2*pi*f0*t - phi);


            case 4 % Swell
                swell_amplitude = 1.2 + 0.4*rand(); % 1.2-1.6 range
                y = swell_amplitude * sin(2*pi*f0*t - phi);

            case 5 % Notch + Transient
                notch_width = 0.005 + 0.003*rand();
                notch_start = 0.05 + 0.05*rand();
                notch = -0.3 * (t > notch_start & t < (notch_start + notch_width));
                transient_freq = 250 + 100*rand();
                transient = 0.8 * exp(-200*t) .* sin(2*pi*transient_freq*t);
                y = sin(2*pi*f0*t - phi) + notch + transient;

            case 6 % Interrupt
                int_start = 0.04 + 0.02*rand();
                int_duration = 0.04 + 0.03*rand();
                y = sin(2*pi*f0*t - phi);
                y(t > int_start & t < (int_start + int_duration)) = 0;

            case 7 % Flicker + Sag
                flicker_freq = 5 + 3*rand();
                flicker_amp = 0.08 + 0.04*rand();
                flicker = flicker_amp * sin(2*pi*flicker_freq*t);
                sag_amplitude = 0.4 + 0.2*rand();
                y = sag_amplitude * sin(2*pi*f0*t - phi) + flicker;

            case 8 % Harmonics
                h3_amp = 0.15 + 0.1*rand();
                h5_amp = 0.08 + 0.05*rand();
                harmonics = h3_amp * sin(3*2*pi*f0*t) + h5_amp * sin(5*2*pi*f0*t);
                y = sin(2*pi*f0*t - phi) + harmonics;

            case 9 % Transient
                transient_start = 0.05 + 0.03*rand();
                transient_freq = 250 + 100*rand();
                transient = 0.8 * exp(-200*(t-transient_start).^2) .* sin(2*pi*transient_freq*t);
                y = sin(2*pi*f0*t - phi) + transient;

            case 10 % Flicker
                flicker_freq = 5 + 3*rand();
                flicker_amp = 0.08 + 0.04*rand();
                flicker = flicker_amp * sin(2*pi*flicker_freq*t);
                y = sin(2*pi*f0*t - phi) + flicker;

            case 11 % Notch
                notch_width = 0.005 + 0.003*rand();
                notch_start = 0.05 + 0.05*rand();
                notch = -0.3 * (t > notch_start & t < (notch_start + notch_width));
                y = sin(2*pi*f0*t - phi) + notch;

            case 12 % Spike
                spike_start = 0.05 + 0.02*rand();
                spike = 1.5 * exp(-300*(t-spike_start).^2) .* sin(2*pi*800*t);
                y = sin(2*pi*f0*t - phi) + spike;

            case 13 % Spike + Harmonics
                h3_amp = 0.15 + 0.1*rand();
                h5_amp = 0.08 + 0.05*rand();
                harmonics = h3_amp * sin(3*2*pi*f0*t) + h5_amp * sin(5*2*pi*f0*t);
                spike_start = 0.05 + 0.02*rand();
                spike = 1.5 * exp(-300*(t-spike_start).^2) .* sin(2*pi*800*t);
                y = sin(2*pi*f0*t - phi) + spike + harmonics;

            case 14 % Sag + Harmonics
                sag_amplitude = 0.4 + 0.2*rand();
                h3_amp = 0.15 + 0.1*rand();
                h5_amp = 0.08 + 0.05*rand();
                harmonics = h3_amp * sin(3*2*pi*f0*t) + h5_amp * sin(5*2*pi*f0*t);
                y = sag_amplitude * sin(2*pi*f0*t - phi) + harmonics;

            case 15 % Swell + Harmonics
                swell_amplitude = 1.2 + 0.4*rand();
                h3_amp = 0.15 + 0.1*rand();
                h5_amp = 0.08 + 0.05*rand();
                harmonics = h3_amp * sin(3*2*pi*f0*t) + h5_amp * sin(5*2*pi*f0*t);
                y = swell_amplitude * sin(2*pi*f0*t - phi) + harmonics;

            case 16 % Interrupt + Harmonics
                int_start = 0.04 + 0.02*rand();
                int_duration = 0.04 + 0.03*rand();
                interrupt = sin(2*pi*f0*t - phi);
                interrupt(t > int_start & t < (int_start + int_duration)) = 0;
                h3_amp = 0.15 + 0.1*rand();
                h5_amp = 0.08 + 0.05*rand();
                harmonics = h3_amp * sin(3*2*pi*f0*t) + h5_amp * sin(5*2*pi*f0*t);
                y = interrupt + harmonics;

            case 17 % Transient + Harmonics
                transient_start = 0.05 + 0.03*rand();
                transient_freq = 250 + 100*rand();
                transient = 0.8 * exp(-200*(t-transient_start).^2) .* sin(2*pi*transient_freq*t);
                h3_amp = 0.15 + 0.1*rand();
                h5_amp = 0.08 + 0.05*rand();
                harmonics = h3_amp * sin(3*2*pi*f0*t) + h5_amp * sin(5*2*pi*f0*t);
                y = sin(2*pi*f0*t - phi) + transient + harmonics;

            case 18 % Flicker + Harmonics
                flicker_freq = 5 + 3*rand();
                flicker_amp = 0.08 + 0.04*rand();
                flicker = flicker_amp * sin(2*pi*flicker_freq*t);
                h3_amp = 0.15 + 0.1*rand();
                h5_amp = 0.08 + 0.05*rand();
                harmonics = h3_amp * sin(3*2*pi*f0*t) + h5_amp * sin(5*2*pi*f0*t);
                y = sin(2*pi*f0*t - phi) + flicker + harmonics;

            case 19 % Sag + Transient
                sag_amplitude = 0.4 + 0.2*rand();
                transient_start = 0.05 + 0.03*rand();
                transient_freq = 250 + 100*rand();
                transient = 0.8 * exp(-200*(t-transient_start).^2) .* sin(2*pi*transient_freq*t);
                y = sag_amplitude * sin(2*pi*f0*t - phi) + transient;
        end
        
        % Add noise
        y = awgn(y, SNR);
        z = [z; y];
        cl = [cl; {class_names{class_idx}}]; % Append class name as cell
    end
end

% Save signals
save('PQD_Signals.mat', 'z', 'cl', 't', 'fs');
disp('Signal generation complete and saved as PQD_Signals.mat');

% Plot example of each disturbance type
figure('Position', [100, 100, 1200, 800]);
for class_idx = 1:length(class_names)
    idx = find(strcmp(cl, class_names{class_idx}), 1);
    subplot(5, 4, class_idx);
    plot(t, z(idx, :));
    title(class_names{class_idx});
    xlim([0 0.2]);
    ylim([-2.5 2.5]);
    grid on;
end

sgtitle('Examples of Power Quality Disturbances');