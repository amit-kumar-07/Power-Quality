%% Part 3: Classification of PQD using Random Forest - 70/30 Split
% Uses EWT-based features extracted in Part 2
clc; clear; close all;

disp('------------------------------------------------------');
disp('POWER QUALITY DISTURBANCE CLASSIFICATION');
disp('Random Forest Classification of EWT-based Features');
disp('70/30 Train/Test Split');
disp('------------------------------------------------------');

% Load extracted EWT features (70/30 split)
load('PQD_features_EWT_70_30.mat');
disp('Loaded EWT feature data (70/30 split)');

% Convert cell array labels to categorical for classification
train_labels_cat = categorical(train_labels);
test_labels_cat = categorical(test_labels);

% Display dataset information
disp(['Number of classes: ', num2str(length(unique(train_labels_cat)))]);
disp(['Training samples: ', num2str(size(train_features_norm, 1))]);
disp(['Test samples: ', num2str(size(test_features_norm, 1))]);

%% Train Random Forest Classifier with optimized hyperparameters
disp('Training Random Forest classifier...');
rng(42, 'twister'); % For reproducibility

% Create template with optimized hyperparameters for higher accuracy
t = templateTree('MinLeafSize', 1, 'MaxNumSplits', 150);

% Train the model - using fitcensemble for CLASSIFICATION
rf_model = fitcensemble(train_features_norm, train_labels_cat, ...
    'Method', 'Bag', ...  % Bagging = Random Forest
    'NumLearningCycles', 300, ... % Increased number of trees for better accuracy
    'Learners', t);

%% Evaluate the model
disp('Evaluating model on test set...');

% Make predictions
[predicted_labels, scores] = predict(rf_model, test_features_norm);

% Calculate accuracy
accuracy = sum(predicted_labels == test_labels_cat) / length(test_labels_cat);
disp(['Test accuracy: ', num2str(accuracy*100, '%.2f'), '%']);

%% Confusion Matrix
% Create confusion matrix
cm = confusionmat(test_labels_cat, predicted_labels);

% Plot confusion matrix
figure('Position', [100, 100, 1000, 800]);
conf_chart = confusionchart(cm, categories(test_labels_cat), 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix for Power Quality Disturbance Classification');
sgtitle(['Random Forest Classification Accuracy: ', num2str(accuracy*100, '%.2f'), '%']);

%% Calculate class-wise metrics
class_names = categories(test_labels_cat);
num_classes = length(class_names);
precision = zeros(num_classes, 1);
recall = zeros(num_classes, 1);
f1_score = zeros(num_classes, 1);

for i = 1:num_classes
    % True positives
    tp = sum((predicted_labels == class_names{i}) & (test_labels_cat == class_names{i}));
    
    % False positives
    fp = sum((predicted_labels == class_names{i}) & (test_labels_cat ~= class_names{i}));
    
    % False negatives
    fn = sum((predicted_labels ~= class_names{i}) & (test_labels_cat == class_names{i}));
    
    % Calculate metrics (avoid division by zero)
    precision(i) = tp / (tp + fp + eps);
    recall(i) = tp / (tp + fn + eps);
    f1_score(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i) + eps);
    
    % Display per-class metrics
    fprintf('Class %s: Precision=%.2f%%, Recall=%.2f%%, F1-Score=%.2f%%\n', ...
        char(class_names{i}), precision(i)*100, recall(i)*100, f1_score(i)*100);
end

% Calculate macro-averaged metrics
macro_precision = mean(precision);
macro_recall = mean(recall);
macro_f1 = mean(f1_score);

disp(' ');
disp(['Macro-averaged Precision: ', num2str(macro_precision*100, '%.2f'), '%']);
disp(['Macro-averaged Recall: ', num2str(macro_recall*100, '%.2f'), '%']);
disp(['Macro-averaged F1-score: ', num2str(macro_f1*100, '%.2f'), '%']);

%% Save Results
results = struct();
results.accuracy = accuracy;
results.precision = precision;
results.recall = recall;
results.f1_score = f1_score;
results.macro_precision = macro_precision;
results.macro_recall = macro_recall;
results.macro_f1 = macro_f1;
results.confusion_matrix = cm;
results.class_names = class_names;
results.predicted_labels = predicted_labels;
results.actual_labels = test_labels_cat;

save('PQD_classification_results_70_30.mat', 'results', 'rf_model');

% Final output summary
disp('------------------------------------------------------');
disp(['FINAL CLASSIFICATION PERFORMANCE:']);
disp(['Accuracy: ', num2str(accuracy*100, '%.2f'), '%']);
disp(['Precision: ', num2str(macro_precision*100, '%.2f'), '%']);
disp(['Recall: ', num2str(macro_recall*100, '%.2f'), '%']);
disp(['F1-score: ', num2str(macro_f1*100, '%.2f'), '%']);
disp('------------------------------------------------------');

%% Feature Importance Analysis
try
    % Get feature importance
    imp = predictorImportance(rf_model);
    
    % Sort features by importance
    [sorted_imp, idx] = sort(imp, 'descend');
    
    % Plot feature importance
    figure;
    bar(sorted_imp(1:min(20, length(sorted_imp))));
    title('Top 20 Most Important Features');
    xlabel('Feature Index');
    ylabel('Importance Score');
    xticks(1:min(20, length(sorted_imp)));
    xticklabels(idx(1:min(20, length(sorted_imp))));
    xtickangle(45);
    grid on;
    
    % Display top 10 important features
    disp('Top 10 Most Important Features:');
    for i = 1:min(10, length(sorted_imp))
        fprintf('%d. Feature %d: %.4f\n', i, idx(i), sorted_imp(i));
    end
catch
    disp('Feature importance calculation not available for this model type.');
end

%% ROC Curves for Selected Classes
try
    figure('Position', [100, 100, 900, 700]);
    
    % Plot ROC curves for first 6 classes (for clarity)
    for i = 1:min(6, num_classes)
        [X, Y, ~, AUC] = perfcurve(double(test_labels_cat == class_names{i}), scores(:, i), true);
        
        subplot(2, 3, i);
        plot(X, Y, 'LineWidth', 2);
        hold on;
        plot([0, 1], [0, 1], 'r--');
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title([char(class_names{i}), ' (AUC = ', num2str(AUC, '%.3f'), ')']);
        grid on;
    end
    
    sgtitle('ROC Curves for Selected Classes');
catch
    disp('ROC curve generation not available for this model configuration.');
end

disp('Classification analysis complete.');