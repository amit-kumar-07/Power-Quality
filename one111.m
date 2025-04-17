%% Part 3: Classification of PQD using Random Forest - Fixed Version
% Uses EWT-based features extracted in Part 2
clc; clear; close all;

disp('------------------------------------------------------');
disp('POWER QUALITY DISTURBANCE CLASSIFICATION');
disp('Random Forest Classification of EWT-based Features');
disp('------------------------------------------------------');

% Load extracted EWT features
load('PQD_features_EWT.mat');
disp('Loaded EWT feature data');

% Convert cell array labels to categorical for classification
train_labels_cat = categorical(train_labels);
val_labels_cat = categorical(val_labels);
test_labels_cat = categorical(test_labels);

% Display dataset information
disp(['Number of classes: ', num2str(length(unique(train_labels_cat)))]);
disp(['Training samples: ', num2str(size(train_features_norm, 1))]);
disp(['Validation samples: ', num2str(size(val_features_norm, 1))]);
disp(['Test samples: ', num2str(size(test_features_norm, 1))]);

%% Create training and testing datasets
% Combine training and validation for final model
X_train = [train_features_norm; val_features_norm];
Y_train = [train_labels_cat; val_labels_cat];
X_test = test_features_norm;
Y_test = test_labels_cat;

%% Train Random Forest Classifier with fixed hyperparameters
disp('Training Random Forest classifier...');
rng(42, 'twister'); % For reproducibility

% Create template with fixed hyperparameters
t = templateTree('MinLeafSize', 1, 'MaxNumSplits', 100);

% Train the model - using fitcensemble for CLASSIFICATION (not fitrensemble)
rf_model = fitcensemble(X_train, Y_train, ...
    'Method', 'Bag', ...  % Bagging = Random Forest
    'NumLearningCycles', 200, ... % Number of trees
    'Learners', t);

%% Evaluate the model
disp('Evaluating model on test set...');

% Make predictions
[predicted_labels, scores] = predict(rf_model, X_test);

% Calculate accuracy
accuracy = sum(predicted_labels == Y_test) / length(Y_test);
disp(['Test accuracy: ', num2str(accuracy*100, '%.2f'), '%']);

%% Confusion Matrix
% Create confusion matrix
cm = confusionmat(Y_test, predicted_labels);

% Plot confusion matrix
figure('Position', [100, 100, 900, 700]);
confusionchart(cm, categories(Y_test), 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix for Power Quality Disturbance Classification');
sgtitle(['Random Forest Classification Accuracy: ', num2str(accuracy*100, '%.2f'), '%']);

%% Calculate class-wise metrics
class_names = categories(Y_test);
num_classes = length(class_names);
precision = zeros(num_classes, 1);
recall = zeros(num_classes, 1);
f1_score = zeros(num_classes, 1);

for i = 1:num_classes
    % True positives
    tp = sum((predicted_labels == class_names{i}) & (Y_test == class_names{i}));
    
    % False positives
    fp = sum((predicted_labels == class_names{i}) & (Y_test ~= class_names{i}));
    
    % False negatives
    fn = sum((predicted_labels ~= class_names{i}) & (Y_test == class_names{i}));
    
    % Calculate metrics (avoid division by zero)
    precision(i) = tp / (tp + fp + eps);
    recall(i) = tp / (tp + fn + eps);
    f1_score(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i) + eps);
end

% Calculate macro-averaged metrics
macro_precision = mean(precision);
macro_recall = mean(recall);
macro_f1 = mean(f1_score);

disp(['Macro-averaged Precision: ', num2str(macro_precision*100, '%.2f'), '%']);
disp(['Macro-averaged Recall: ', num2str(macro_recall*100, '%.2f'), '%']);
disp(['Macro-averaged F1-score: ', num2str(macro_f1*100, '%.2f'), '%']);

%% Save Results
results = struct();
results.accuracy = accuracy;
results.precision = macro_precision;
results.recall = macro_recall;
results.f1_score = macro_f1;
results.confusion_matrix = cm;
results.class_names = class_names;
results.predicted_labels = predicted_labels;
results.actual_labels = Y_test;

save('PQD_classification_results.mat', 'results', 'rf_model');

% Final output summary
disp('------------------------------------------------------');
disp(['FINAL CLASSIFICATION PERFORMANCE:']);
disp(['Accuracy: ', num2str(accuracy*100, '%.2f'), '%']);
disp(['Precision: ', num2str(macro_precision*100, '%.2f'), '%']);
disp(['Recall: ', num2str(macro_recall*100, '%.2f'), '%']);
disp(['F1-score: ', num2str(macro_f1*100, '%.2f'), '%']);
disp('------------------------------------------------------');

%% Feature Importance (if available)
try
    imp = predictorImportance(rf_model);
    
    % Plot feature importance
    figure;
    bar(imp);
    title('Feature Importance');
    xlabel('Feature Index');
    ylabel('Importance Score');
    grid on;
catch
    disp('Feature importance calculation not available for this model type.');
end