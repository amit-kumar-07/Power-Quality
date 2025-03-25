%% Part 3: Optimized Power Quality Classifier (Modified Confusion Matrix)
clc; clear; close all;

% Load extracted features
load('PQD_features_final.mat');

% Convert labels to categorical
train_labels = categorical(train_labels);
val_labels = categorical(val_labels);
test_labels = categorical(test_labels);

%% Data Preparation
X_train = train_features_norm;
Y_train = train_labels;
X_val = val_features_norm;
Y_val = val_labels;
X_test = test_features_norm;
Y_test = test_labels;
class_names = categories(Y_train);

%% Handle Feature Dimension Mismatch
num_data_features = size(X_train, 2);
num_feature_names = length(feature_names);

% Adjust feature names if mismatch exists
if num_data_features ~= num_feature_names
    fprintf('Adjusting feature names to match data dimensions...\n');
    fprintf('Original: %d feature names, Data has %d features\n', num_feature_names, num_data_features);
    
    if num_data_features > num_feature_names
        % Add missing feature names
        for i = num_feature_names+1:num_data_features
            feature_names{end+1} = sprintf('Feature_%d', i);
        end
    else
        % Truncate extra feature names
        feature_names = feature_names(1:num_data_features);
    end
    fprintf('Adjusted to %d feature names\n', length(feature_names));
end

%% Feature Selection using MRMR
[rankedIdx, ~] = fscmrmr(X_train, Y_train);
num_features_to_select = min(60, num_data_features);
selected_feats = rankedIdx(1:num_features_to_select);

% Apply feature selection
X_train = X_train(:, selected_feats);
X_val = X_val(:, selected_feats);
X_test = X_test(:, selected_feats);
feature_names = feature_names(selected_feats);

%% Train Optimized Random Forest
rng(42); % For reproducibility

model = TreeBagger(350, X_train, Y_train, ...
                  'Method', 'classification', ...
                  'MinLeafSize', 2, ...
                  'NumPredictorsToSample', ceil(size(X_train, 2)*0.8), ...
                  'OOBPredictorImportance', 'on', ...
                  'ClassNames', class_names);

%% Model Evaluation
[test_pred, ~] = predict(model, X_test);
test_pred = categorical(test_pred);
test_accuracy = mean(test_pred == Y_test) * 100;

%% Performance Analysis
C = confusionmat(Y_test, test_pred);

% Create custom display matrix
C_display = nan(size(C)); % Initialize with NaN (will display as white)
C_display(logical(eye(size(C)))) = diag(C); % Set diagonal elements

% Find non-zero off-diagonal elements
[row, col] = find(C & ~eye(size(C)));
for i = 1:length(row)
    C_display(row(i), col(i)) = C(row(i), col(i));
end

% Visual Confusion Matrix with custom display
figure('Position', [100, 100, 800, 700]);
h = heatmap(class_names, class_names, C_display, ...
           'Colormap', parula, ...
           'ColorScaling', 'scaled', ...
           'MissingDataColor', [1 1 1], ... % White for NaN values
           'FontSize', 12);
h.Title = ['Confusion Matrix (Accuracy: ' sprintf('%.2f', test_accuracy) '%)'];
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';

% Calculate metrics
precision = diag(C)./sum(C,1)';
recall = diag(C)./sum(C,2);
f1_score = 2*(precision.*recall)./(precision+recall);

% Display results
fprintf('\n=== Model Performance ===\n');
fprintf('Test Accuracy: %.2f%%\n\n', test_accuracy);
disp('Confusion Matrix Counts:');
disp(array2table(C, 'RowNames', class_names, 'VariableNames', class_names));

disp('Class-wise Metrics:');
metrics = table(precision, recall, f1_score, ...
               'RowNames', class_names, ...
               'VariableNames', {'Precision','Recall','F1_Score'});
disp(metrics);

%% Feature Importance
imp = model.OOBPermutedPredictorDeltaError;
[~, idx] = sort(imp, 'descend');
top_features = feature_names(idx(1:min(10,end)));

figure('Position', [100, 100, 900, 600]);
barh(imp(idx(1:10)));
set(gca, 'YTickLabel', top_features, 'YTick', 1:length(top_features));
title('Top 10 Important Features');
xlabel('Importance Score');
grid on;

%% Save Model
save('PQD_final_model.mat', 'model', 'test_accuracy', 'feature_names');
fprintf('Model saved to PQD_final_model.mat\n');