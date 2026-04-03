%% SVM STROKE PREDICTION MODEL WITH 5-FOLD CV


fprintf('=== SVM STROKE PREDICTION MODEL WITH 5-FOLD CROSS-VALIDATION ===\n');

% Data Preprocessing
Data2_clean = rmmissing(Data2);
fprintf('After removing missing values: %d rows remaining\n', size(Data2_clean, 1));

% Create processed data table
processed_data = table();
processed_data.age = Data2_clean.age;
processed_data.hypertension = Data2_clean.hypertension;
processed_data.heart_disease = Data2_clean.heart_disease;
if ismember('avg_glucose_level', Data2_clean.Properties.VariableNames)
    processed_data.avg_glucose_level = Data2_clean.avg_glucose_level;
end
if ismember('bmi', Data2_clean.Properties.VariableNames)
    processed_data.bmi = Data2_clean.bmi;
end
processed_data.stroke = Data2_clean.stroke;

% Convert categorical variables
if ismember('gender', Data2_clean.Properties.VariableNames)
    processed_data.gender_numeric = grp2idx(Data2_clean.gender);
end
if ismember('ever_married', Data2_clean.Properties.VariableNames)
    processed_data.ever_married_numeric = grp2idx(Data2_clean.ever_married);
end
if ismember('work_type', Data2_clean.Properties.VariableNames)
    processed_data.work_type_numeric = grp2idx(Data2_clean.work_type);
end
if ismember('Residence_type', Data2_clean.Properties.VariableNames)
    processed_data.residence_type_numeric = grp2idx(Data2_clean.Residence_type);
end
if ismember('smoking_status', Data2_clean.Properties.VariableNames)
    processed_data.smoking_status_numeric = grp2idx(Data2_clean.smoking_status);
end

% Select predictors
predictor_vars = {'age', 'hypertension', 'heart_disease', 'ever_married_numeric', ...
                  'work_type_numeric', 'residence_type_numeric', 'avg_glucose_level', ...
                  'bmi', 'smoking_status_numeric', 'gender_numeric'};
available_vars = processed_data.Properties.VariableNames;
use_vars = predictor_vars(ismember(predictor_vars, available_vars));

X = processed_data{:, use_vars};
y = processed_data.stroke;

fprintf('\nUsing %d predictors\n', length(use_vars));
for i = 1:length(use_vars)
    fprintf('  %d. %s\n', i, use_vars{i});
end

% Check class distribution
fprintf('\nClass Distribution:\n');
fprintf('  Stroke (1): %d samples (%.1f%%)\n', sum(y==1), mean(y==1)*100);
fprintf('  No Stroke (0): %d samples (%.1f%%)\n', sum(y==0), mean(y==0)*100);

stroke_cases = sum(y==1);
non_stroke_cases = sum(y==0);
total_cases = length(y);
imbalance_ratio = non_stroke_cases / stroke_cases;

fprintf('\nImbalance ratio: %.2f:1 (non-stroke:stroke)\n', imbalance_ratio);


%% 5-FOLD CROSS-VALIDATION SETUP

fprintf('\n=== 5-FOLD CROSS-VALIDATION SETUP ===\n');

% Set random seed for reproducibility
rng(42);

% Create 5-fold partition
cv = cvpartition(y, 'KFold', 5);

% Initialize arrays to store results
all_y_true = [];
all_y_pred = [];
all_y_scores = [];
fold_accuracies = zeros(5, 1);
fold_precisions = zeros(5, 1);
fold_recalls = zeros(5, 1);
fold_f1s = zeros(5, 1);
fold_aucs = zeros(5, 1);
fold_thresholds = zeros(5, 1);

% Feature importance storage
feature_weights = zeros(length(use_vars), 5);

% Initialize cell arrays for ROC curve storage
all_fpr = cell(5, 1);
all_tpr = cell(5, 1);
all_aucs_folds = zeros(5, 1);

% For final visualization, also keep a holdout set
rng(42);
cv_final = cvpartition(y, 'Holdout', 0.3);
train_idx_final = training(cv_final);
test_idx_final = test(cv_final);

X_train_final = X(train_idx_final, :);
y_train_final = y(train_idx_final);
X_test_final = X(test_idx_final, :);
y_test_final = y(test_idx_final);

fprintf('Final model will use: %d training, %d testing samples\n', ...
    length(y_train_final), length(y_test_final));


%% TRAIN AND EVALUATE SVM FOR EACH FOLD

fprintf('\nTraining and evaluating SVM model with 5-fold CV...\n');

for fold = 1:5
    fprintf('\n--- Fold %d/%d ---\n', fold, 5);
    
    % Get training and test indices
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);
    
    % Split data
    X_train = X(train_idx, :);
    y_train = y(train_idx);
    X_test = X(test_idx, :);
    y_test = y(test_idx);
    
    % Standardize features
    [X_train_scaled, mu, sigma] = zscore(X_train);
    X_test_scaled = (X_test - mu) ./ sigma;
    
    % Handle any NaN from standardization
    X_train_scaled(isnan(X_train_scaled)) = 0;
    X_test_scaled(isnan(X_test_scaled)) = 0;
    
    % Calculate class distribution
    n_pos = sum(y_train == 1);
    n_neg = sum(y_train == 0);
    pos_ratio = n_pos / (n_pos + n_neg);
    
    fprintf('  Class distribution: %d positive (%.1f%%), %d negative (%.1f%%)\n', ...
        n_pos, pos_ratio*100, n_neg, (1-pos_ratio)*100);
    
    % Train SVM model with proper imbalance handling
    fprintf('  Training SVM... ');
    
    try
        % Calculate appropriate cost for imbalanced data
        cost_ratio = n_neg / n_pos;  % e.g., 95/5 = 19
        
        % Create weights vector
        weights = ones(size(y_train));
        weights(y_train == 1) = cost_ratio;  % Higher weight for minority class
        weights(y_train == 0) = 1;
        
        % Train SVM with weighted samples
        svm_model = fitcsvm(X_train_scaled, y_train, ...
            'KernelFunction', 'linear', ...
            'Standardize', false, ...  % Already standardized
            'ClassNames', [0, 1], ...
            'Weights', weights, ...
            'BoxConstraint', 1, ...
            'Verbose', 0);
        
        % Get scores (distance from hyperplane)
        [~, scores] = predict(svm_model, X_test_scaled);
        y_scores_raw = scores(:, 2);  % Raw SVM scores
        
        % Convert SVM scores to probabilities using sigmoid transformation
        y_scores = 1 ./ (1 + exp(-y_scores_raw));
        
        % Store feature weights
        if isprop(svm_model, 'Beta')
            feature_weights(:, fold) = svm_model.Beta;
        end
        
    catch ME
        % Fallback method
        fprintf('\n  Using simpler SVM approach... ');
        
        % Try fitclinear with explicit cost matrix
        cost_matrix = [0, 1;         % Cost for class 0
                      cost_ratio, 0]; % Higher cost for class 1
        
        svm_model = fitclinear(X_train_scaled, y_train, ...
            'Learner', 'svm', ...
            'ClassNames', [0, 1], ...
            'Cost', cost_matrix, ...
            'Verbose', 0);
        
        % Get scores
        [~, scores] = predict(svm_model, X_test_scaled);
        y_scores = scores(:, 2);
        
        % Store feature weights
        if isprop(svm_model, 'Beta')
            feature_weights(:, fold) = svm_model.Beta;
        end
    end
    
    % Find optimal threshold
    thresholds = linspace(0, 1, 101);  % For probabilities
    best_f1 = 0;
    optimal_threshold = 0.5;  % Default for probabilities
    best_y_pred = [];
    
    for t = thresholds
        y_pred_temp = double(y_scores >= t);
        
        % Skip if all predictions are the same
        if all(y_pred_temp == 0) || all(y_pred_temp == 1)
            continue;
        end
        
        [~, prec_temp, rec_temp, f1_temp] = calculate_metrics(y_test, y_pred_temp);
        
        if f1_temp > best_f1
            best_f1 = f1_temp;
            optimal_threshold = t;
            best_y_pred = y_pred_temp;
        end
    end
    
    % If no good threshold found, use percentile-based
    if isempty(best_y_pred)
        expected_pos_rate = pos_ratio;
        percentile = (1 - expected_pos_rate) * 100;
        
        if percentile > 100, percentile = 99; end
        if percentile < 0, percentile = 1; end
        
        optimal_threshold = prctile(y_scores, percentile);
        best_y_pred = double(y_scores >= optimal_threshold);
    end
    
    y_pred = best_y_pred;
    
    % Calculate metrics
    [fold_acc, fold_prec, fold_rec, fold_f1] = calculate_metrics(y_test, y_pred);
    
    % Calculate AUC and store ROC curve
    [fpr_fold, tpr_fold, ~, fold_auc] = perfcurve(y_test, y_scores, 1);
    
    % Store ROC curve data
    all_fpr{fold} = fpr_fold;
    all_tpr{fold} = tpr_fold;
    all_aucs_folds(fold) = fold_auc;
    
    % Store results
    fold_accuracies(fold) = fold_acc;
    fold_precisions(fold) = fold_prec;
    fold_recalls(fold) = fold_rec;
    fold_f1s(fold) = fold_f1;
    fold_aucs(fold) = fold_auc;
    fold_thresholds(fold) = optimal_threshold;
    
    % Store for overall evaluation
    all_y_true = [all_y_true; y_test];
    all_y_pred = [all_y_pred; y_pred];
    all_y_scores = [all_y_scores; y_scores];
    
    fprintf(' Done\n');
    fprintf('  Threshold used: %.3f\n', optimal_threshold);
    fprintf('  Predictions: %d pos (%.1f%%), %d neg (%.1f%%)\n', ...
        sum(y_pred == 1), mean(y_pred == 1)*100, sum(y_pred == 0), mean(y_pred == 0)*100);
    fprintf('  Accuracy: %.2f%%, Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%, AUC: %.3f\n', ...
        fold_acc*100, fold_prec*100, fold_rec*100, fold_f1*100, fold_auc);
end


%% TRAIN FINAL MODEL FOR VISUALIZATION

fprintf('\n=== TRAINING FINAL MODEL FOR VISUALIZATION ===\n');

% Standardize features for final model
[X_train_final_scaled, mu_final, sigma_final] = zscore(X_train_final);
X_test_final_scaled = (X_test_final - mu_final) ./ sigma_final;

% Handle any NaN
X_train_final_scaled(isnan(X_train_final_scaled)) = 0;
X_test_final_scaled(isnan(X_test_final_scaled)) = 0;

% Calculate class weights for final model
n_pos_final = sum(y_train_final == 1);
n_neg_final = sum(y_train_final == 0);
cost_ratio_final = n_neg_final / n_pos_final;

weights_final = ones(size(y_train_final));
weights_final(y_train_final == 1) = cost_ratio_final;
weights_final(y_train_final == 0) = 1;

% Train final SVM model
svm_model_final = fitcsvm(X_train_final_scaled, y_train_final, ...
    'KernelFunction', 'linear', ...
    'Standardize', false, ...
    'ClassNames', [0, 1], ...
    'Weights', weights_final, ...
    'BoxConstraint', 1, ...
    'Verbose', 0);

% Get predictions and scores
[~, scores_final] = predict(svm_model_final, X_test_final_scaled);
y_scores_final_raw = scores_final(:, 2);
y_scores_final = 1 ./ (1 + exp(-y_scores_final_raw));  % Sigmoid transformation

% Find optimal threshold for final model
thresholds = linspace(0, 1, 101);
best_f1_final = 0;
optimal_threshold_final = 0.5;
best_y_pred_final = [];

for t = thresholds
    y_pred_temp = double(y_scores_final >= t);
    
    if all(y_pred_temp == 0) || all(y_pred_temp == 1)
        continue;
    end
    
    [~, prec_temp, rec_temp, f1_temp] = calculate_metrics(y_test_final, y_pred_temp);
    
    if f1_temp > best_f1_final
        best_f1_final = f1_temp;
        optimal_threshold_final = t;
        best_y_pred_final = y_pred_temp;
    end
end

y_pred_final = best_y_pred_final;

% Calculate metrics for final model
[accuracy_final, precision_final, recall_final, f1_final] = calculate_metrics(y_test_final, y_pred_final);

% Calculate ROC and AUC for final model
[X_roc_final, Y_roc_final, T_final, AUC_final] = perfcurve(y_test_final, y_scores_final, 1);

% Calculate Precision-Recall curve and AUPRC
[X_pr_final, Y_pr_final, ~, AUPRC_final] = perfcurve(y_test_final, y_scores_final, 1, 'XCrit', 'reca', 'YCrit', 'prec');

fprintf('\nFinal Model Results:\n');
fprintf('  Threshold: %.3f\n', optimal_threshold_final);
fprintf('  Accuracy:  %.2f%%\n', accuracy_final*100);
fprintf('  Precision: %.2f%%\n', precision_final*100);
fprintf('  Recall:    %.2f%%\n', recall_final*100);
fprintf('  F1-Score:  %.2f%%\n', f1_final*100);
fprintf('  AUC:       %.3f\n', AUC_final);
fprintf('  AUPRC:     %.3f\n', AUPRC_final);

%% CROSS-VALIDATION RESULTS SUMMARY


fprintf('\n=== 5-FOLD CROSS-VALIDATION RESULTS ===\n');
fprintf('Average across 5 folds:\n');
fprintf('  Accuracy:  %.2f%% ± %.2f%%\n', mean(fold_accuracies)*100, std(fold_accuracies)*100);
fprintf('  Precision: %.2f%% ± %.2f%%\n', mean(fold_precisions)*100, std(fold_precisions)*100);
fprintf('  Recall:    %.2f%% ± %.2f%%\n', mean(fold_recalls)*100, std(fold_recalls)*100);
fprintf('  F1-score:  %.2f%% ± %.2f%%\n', mean(fold_f1s)*100, std(fold_f1s)*100);
fprintf('  AUC:       %.3f ± %.3f\n', mean(fold_aucs), std(fold_aucs));

% Find best fold based on AUC
[best_auc, best_fold] = max(fold_aucs);
fprintf('\nBest performing fold: Fold %d (AUC = %.3f)\n', best_fold, best_auc);

%% PLOT

fprintf('\n=== GENERATING VISUALIZATIONS ===\n');

% Main 2x3 plot grid
figure;
sgtitle('Support Vector Machine for Stroke Prediction (5-Fold CV)', 'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: ROC Curve with optimal threshold
subplot(2,3,1);
plot(X_roc_final, Y_roc_final, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
% Find the point on ROC curve closest to our threshold
[~, roc_point_idx] = min(abs(T_final - optimal_threshold_final));
plot(X_roc_final(roc_point_idx), Y_roc_final(roc_point_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
xlabel('False Positive Rate');
ylabel('True Positive Rate (Sensitivity)');
title(sprintf('ROC Curve (AUC = %.4f)', AUC_final));
legend(sprintf('SVM (AUC=%.4f)', AUC_final), ...
       'Random Classifier', ...
       sprintf('Threshold=%.3f', optimal_threshold_final), ...
       'Location', 'southeast');
grid on;

% Add CV performance annotation
text(0.6, 0.2, sprintf('CV AUC: %.3f ± %.3f', mean(fold_aucs), std(fold_aucs)), ...
     'FontSize', 9, 'BackgroundColor', 'white', 'EdgeColor', 'blue', 'Margin', 2);

% Plot 2: Precision-Recall Curve
subplot(2,3,2);
plot(X_pr_final, Y_pr_final, 'g-', 'LineWidth', 2);
xlabel('Recall (Sensitivity)');
ylabel('Precision');
title(sprintf('Precision-Recall Curve (AUPRC = %.4f)', AUPRC_final));
grid on;
% Add baseline for imbalanced data
baseline_precision = sum(y_test_final) / length(y_test_final);
line([0 1], [baseline_precision baseline_precision], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 1);
legend('SVM', sprintf('Baseline (%.4f)', baseline_precision), 'Location', 'southwest');

% Plot 3: Probability distributions by class
subplot(2,3,3);
hold on;
if sum(y_test_final==0) > 0
    histogram(y_scores_final(y_test_final==0), 'BinWidth', 0.02, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
if sum(y_test_final==1) > 0
    histogram(y_scores_final(y_test_final==1), 'BinWidth', 0.02, 'FaceColor', 'red', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
line([optimal_threshold_final optimal_threshold_final], ylim(), 'Color', 'black', 'LineWidth', 2, 'LineStyle', '--');
xlabel('Predicted Probability');
ylabel('Density');
title('Probability Distribution by Class (Test Set)');
legend('No Stroke', 'Stroke', sprintf('Threshold=%.3f', optimal_threshold_final), 'Location', 'northwest');
grid on;

% Plot 4: Feature Importance
subplot(2,3,4);
% Calculate average absolute weights across folds
avg_weights = mean(abs(feature_weights), 2);

% Sort features by importance
[sorted_weights, idx] = sort(avg_weights, 'descend');
sorted_names = use_vars(idx);
top_features_to_show = min(8, length(use_vars));

barh(sorted_weights(1:top_features_to_show));
set(gca, 'YTick', 1:top_features_to_show, 'YTickLabel', sorted_names(1:top_features_to_show));
xlabel('Feature Importance (Avg Absolute Weight)');
title('SVM Feature Importance');
grid on;
hold on;
plot([0 0], ylim(), 'r--', 'LineWidth', 2);
legend('Feature Importance', 'No Effect', 'Location', 'southeast');

% Plot 5: Threshold analysis
subplot(2,3,5);
thresholds = 0.01:0.01:0.99;
f1_scores_plot = zeros(size(thresholds));
for i = 1:length(thresholds)
    y_pred_temp = y_scores_final >= thresholds(i);
    TP = sum(y_pred_temp & y_test_final);
    FP = sum(y_pred_temp & ~y_test_final);
    FN = sum(~y_pred_temp & y_test_final);
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    f1_scores_plot(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
end
plot(thresholds, f1_scores_plot, 'k-', 'LineWidth', 2);
hold on;

% Find the index corresponding to the optimal threshold
threshold_index = find(thresholds >= optimal_threshold_final, 1);
if ~isempty(threshold_index)
    plot(optimal_threshold_final, f1_scores_plot(threshold_index), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
else
    % Fallback: use the closest threshold
    [~, closest_idx] = min(abs(thresholds - optimal_threshold_final));
    plot(optimal_threshold_final, f1_scores_plot(closest_idx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
end

xlabel('Classification Threshold');
ylabel('F1-Score');
title('F1-Score vs Threshold');
grid on;
legend('F1-Score', sprintf('Optimal (%.3f)', optimal_threshold_final), 'Location', 'southeast');

% Add CV thresholds from each fold
for fold = 1:5
    fold_threshold = fold_thresholds(fold);
    if fold_threshold >= min(thresholds) && fold_threshold <= max(thresholds)
        idx = find(thresholds >= fold_threshold, 1);
        if ~isempty(idx)
            plot(fold_threshold, f1_scores_plot(idx), 'bx', 'MarkerSize', 6, 'LineWidth', 1);
        end
    end
end

% Plot 6: Calibration plot
subplot(2,3,6);
bin_edges = 0:0.1:1;
bin_centers = 0.05:0.1:0.95;
mean_predicted = zeros(size(bin_centers));
actual_proportion = zeros(size(bin_centers));
for i = 1:length(bin_centers)
    in_bin = y_scores_final >= bin_edges(i) & y_scores_final < bin_edges(i+1);
    if sum(in_bin) > 0
        mean_predicted(i) = mean(y_scores_final(in_bin));
        actual_proportion(i) = mean(y_test_final(in_bin));
    end
end
plot(mean_predicted, actual_proportion, 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 2);
xlabel('Mean Predicted Probability');
ylabel('Actual Proportion');
title('Probability Calibration Plot (Test Set)');
legend('SVM', 'Perfect Calibration', 'Location', 'northwest');
grid on;

% Add number of samples per bin
for i = 1:length(bin_centers)
    in_bin = y_scores_final >= bin_edges(i) & y_scores_final < bin_edges(i+1);
    if sum(in_bin) > 0
        text(mean_predicted(i), actual_proportion(i), ...
             sprintf('n=%d', sum(in_bin)), ...
             'FontSize', 7, 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom');
    end
end

%% ADDITIONAL PLOTS
% Plot 7: Confusion Matrix
figure;
sgtitle('Support Vector Machine for Stroke Prediction (5-Fold CV)', 'FontSize', 16, 'FontWeight', 'bold');
subplot(1,2,1);
y_numeric = double(y_test_final);
y_pred_binary_numeric = double(y_pred_final);
CM_svm = confusionmat(y_numeric, y_pred_binary_numeric);
confusionchart(CM_svm, {'No Stroke', 'Stroke'}, 'Title', 'Confusion Matrix (Optimal Threshold)');

% Plot 8: Class distribution
subplot(1,2,2);
pie([non_stroke_cases, stroke_cases], {'No Stroke', 'Stroke'});
title(sprintf('Class Distribution\n(Imbalance: %.1f:1)', imbalance_ratio));

% Plot 9: Jittered scatter plot for binary classification
figure;
subplot(1,2,1);
% Add small random noise to actual values for visualization
rng(42); % For reproducibility
y_jittered = y_numeric + 0.05 * randn(size(y_numeric)); % Small jitter to actual values
y_pred_jittered = y_scores_final + 0.01 * randn(size(y_scores_final)); % Very small jitter to predictions

% Color points by actual class
colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980]; % Blue and orange
scatter(y_jittered(y_numeric==0), y_pred_jittered(y_numeric==0), 40, colors(1,:), 'filled', ...
        'MarkerFaceAlpha', 0.6, 'DisplayName', 'Actual: No Stroke');
hold on;
scatter(y_jittered(y_numeric==1), y_pred_jittered(y_numeric==1), 40, colors(2,:), 'filled', ...
        'MarkerFaceAlpha', 0.6, 'DisplayName', 'Actual: Stroke');

% Add perfect prediction lines for binary classification
plot([0.5 0.5], [-0.1 1.1], 'r--', 'LineWidth', 2, 'HandleVisibility', 'off');

xlim([-0.2, 1.2]);
ylim([-0.1, 1.1]);
xlabel('Actual Stroke (with jitter)');
ylabel('Predicted Probability (with jitter)');
title(sprintf('Jittered Scatter Plot\n(Optimal Threshold = %.3f)', optimal_threshold_final));
grid on;
legend('Location', 'northwest');

% Plot 10: Box plot of predictions by actual class
subplot(1,2,2);
boxplot(y_scores_final, y_numeric, 'Labels', {'No Stroke', 'Stroke'});
xlabel('Actual Class');
ylabel('Predicted Probability');
title('Box Plot of Predicted Probabilities by Actual Class');
grid on;

% Add optimal threshold line to box plot
hold on;
y_limits = ylim();
plot(xlim(), [optimal_threshold_final optimal_threshold_final], 'r--', 'LineWidth', 2, ...
     'DisplayName', sprintf('Optimal Threshold (%.3f)', optimal_threshold_final));
legend('Location', 'northwest');

%% CROSS-VALIDATION SPECIFIC VISUALIZATION
figure;
sgtitle('5-Fold Cross-Validation Analysis - SVM', 'FontSize', 16, 'FontWeight', 'bold');

% Plot CV performance across folds
subplot(2,3,1);
metrics = {'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'};
fold_metrics_matrix = [fold_accuracies'; fold_precisions'; fold_recalls'; fold_f1s'; fold_aucs'];

plot(1:5, fold_metrics_matrix, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Fold Number');
ylabel('Score');
title('Performance Metrics Across Folds');
legend(metrics, 'Location', 'best');
grid on;
xlim([0.5, 5.5]);

% Plot optimal thresholds across folds
subplot(2,3,2);
bar(1:5, fold_thresholds);
hold on;
plot(xlim(), [optimal_threshold_final optimal_threshold_final], 'r--', 'LineWidth', 2);
xlabel('Fold Number');
ylabel('Optimal Threshold');
title('Optimal Thresholds Across Folds');
grid on;
xticks(1:5);
legend('Fold Threshold', 'Final Model Threshold', 'Location', 'best');

% Plot ROC curves from all folds
subplot(2,3,3);
hold on;
colors = lines(5);
for fold = 1:5
    plot(all_fpr{fold}, all_tpr{fold}, ...
         'Color', colors(fold,:), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Fold %d (AUC=%.3f)', fold, fold_aucs(fold)));
end
plot(X_roc_final, Y_roc_final, 'k-', 'LineWidth', 3, ...
     'DisplayName', sprintf('Final Model (AUC=%.3f)', AUC_final));
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5, 'DisplayName', 'Random');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves: All Folds vs Final Model');
legend('Location', 'southeast', 'FontSize', 8);
grid on;

% Plot feature stability across folds
subplot(2,3,4);
% Calculate coefficient of variation for each feature
feature_stds = std(abs(feature_weights), 0, 2);
feature_means = mean(abs(feature_weights), 2);
feature_cv = feature_stds ./ feature_means;
[~, idx_cv] = sort(feature_cv, 'descend');
n_top_cv = min(8, length(use_vars));

barh(1:n_top_cv, feature_cv(idx_cv(1:n_top_cv)));
set(gca, 'YTick', 1:n_top_cv, 'YTickLabel', use_vars(idx_cv(1:n_top_cv)));
xlabel('Coefficient of Variation');
title('Feature Weight Stability (CV)');
grid on;

% Plot SVM margin visualization 
subplot(2,3,5);
% Get top 2 features by average weight
[~, top2_idx] = sort(avg_weights, 'descend');
if length(top2_idx) >= 2
    top2_features = top2_idx(1:2);
    
    % Create mesh grid for visualization
    x_min = min(X_test_final_scaled(:, top2_features(1))) - 1;
    x_max = max(X_test_final_scaled(:, top2_features(1))) + 1;
    y_min = min(X_test_final_scaled(:, top2_features(2))) - 1;
    y_max = max(X_test_final_scaled(:, top2_features(2))) + 1;
    
    [xx, yy] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));
    
    % Create dummy data with only top 2 features
    dummy_data = zeros(size(xx(:),1), size(X_test_final_scaled, 2));
    dummy_data(:, top2_features(1)) = xx(:);
    dummy_data(:, top2_features(2)) = yy(:);
    
    % Predict
    [~, dummy_scores] = predict(svm_model_final, dummy_data);
    dummy_probs = 1 ./ (1 + exp(-dummy_scores(:,2)));
    zz = reshape(dummy_probs, size(xx));
    
    % Plot decision boundary
    contourf(xx, yy, zz, [optimal_threshold_final, optimal_threshold_final], 'k', 'LineWidth', 2);
    hold on;
    
    % Plot test points
    scatter(X_test_final_scaled(y_test_final==0, top2_features(1)), ...
            X_test_final_scaled(y_test_final==0, top2_features(2)), 30, 'b', 'filled', ...
            'MarkerFaceAlpha', 0.6, 'DisplayName', 'No Stroke');
    scatter(X_test_final_scaled(y_test_final==1, top2_features(1)), ...
            X_test_final_scaled(y_test_final==1, top2_features(2)), 30, 'r', 'filled', ...
            'MarkerFaceAlpha', 0.6, 'DisplayName', 'Stroke');
    
    xlabel(use_vars{top2_features(1)});
    ylabel(use_vars{top2_features(2)});
    title('SVM Decision Boundary (Top 2 Features)');
    legend('Location', 'best');
    grid on;
    colorbar;
end

% Plot performance comparison: CV vs Final Model
subplot(2,3,6);
metrics_final_val = [accuracy_final, precision_final, recall_final, f1_final, AUC_final];
metrics_cv_mean = [mean(fold_accuracies), mean(fold_precisions), mean(fold_recalls), mean(fold_f1s), mean(fold_aucs)];
metrics_cv_std = [std(fold_accuracies), std(fold_precisions), std(fold_recalls), std(fold_f1s), std(fold_aucs)];

x_pos = 1:length(metrics);
bar(x_pos, metrics_final_val, 0.3, 'FaceColor', 'b', 'FaceAlpha', 0.7);
hold on;
errorbar(x_pos, metrics_cv_mean, metrics_cv_std, 'o', 'Color', 'r', ...
         'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
set(gca, 'XTick', x_pos, 'XTickLabel', metrics);
ylabel('Score');
title('Performance: Final Model vs CV (Mean ± SD)');
legend('Final Model', 'CV Mean ± SD', 'Location', 'southwest');
grid on;
rotateXLabels(gca, 45);

%% FINAL PERFORMANCE METRICS

fprintf('\n=== FINAL MODEL PERFORMANCE (TEST SET) ===\n');

% Calculate additional metrics for final model
TP_final = sum(y_pred_final & y_test_final);
FP_final = sum(y_pred_final & ~y_test_final);
TN_final = sum(~y_pred_final & ~y_test_final);
FN_final = sum(~y_pred_final & y_test_final);

specificity_final = TN_final / (TN_final + FP_final + eps);

fprintf('AUC:         %.4f\n', AUC_final);
fprintf('AUPRC:       %.4f\n', AUPRC_final);
fprintf('Accuracy:    %.4f (%.2f%%)\n', accuracy_final, accuracy_final*100);
fprintf('Precision:   %.4f (%.2f%%)\n', precision_final, precision_final*100);
fprintf('Recall:      %.4f (%.2f%%)\n', recall_final, recall_final*100);
fprintf('Specificity: %.4f (%.2f%%)\n', specificity_final, specificity_final*100);
fprintf('F1-Score:    %.4f\n', f1_final);

%% Confusion Matrix Details
fprintf('\n=== CONFUSION MATRIX (Optimal Threshold = %.3f) ===\n', optimal_threshold_final);
fprintf('           Predicted 0   Predicted 1\n');
fprintf('Actual 0:   %6d (TN)    %6d (FP)\n', TN_final, FP_final);
fprintf('Actual 1:   %6d (FN)    %6d (TP)\n', FN_final, TP_final);

%% Feature Importance Analysis
fprintf('\n=== TOP FEATURE IMPORTANCE ===\n');
fprintf('Top 5 most important features:\n');
for i = 1:min(5, length(sorted_names))
    fprintf('%d. %-30s (importance: %.4f)\n', i, sorted_names{i}, sorted_weights(i));
end

%% CROSS-VALIDATION INTERPRETATION
fprintf('\n=== CROSS-VALIDATION INTERPRETATION ===\n');
fprintf('Model Consistency Assessment:\n');
fprintf('  Accuracy CV:  %.2f%% (lower is better)\n', (std(fold_accuracies)/mean(fold_accuracies))*100);
fprintf('  AUC CV:       %.2f%% (lower is better)\n', (std(fold_aucs)/mean(fold_aucs))*100);

if (std(fold_aucs)/mean(fold_aucs)) < 0.1
    fprintf('✓ Model shows good consistency across folds\n');
elseif (std(fold_aucs)/mean(fold_aucs)) < 0.2
    fprintf('⚠️  Model shows moderate consistency across folds\n');
else
    fprintf('⚠️  Model shows high variability across folds\n');
end

fprintf('\nCross-validation advantages demonstrated:\n');
fprintf('1. More reliable performance estimate\n');
fprintf('2. Feature weight stability analysis\n');
fprintf('3. Model stability assessment\n');
fprintf('4. Reduced overfitting risk\n');

fprintf('\n=== SVM WITH 5-FOLD CROSS-VALIDATION ANALYSIS COMPLETE ===\n');

%% HELPER FUNCTION FOR METRICS CALCULATION


function [accuracy, precision, recall, f1] = calculate_metrics(y_true, y_pred)
    % Calculate confusion matrix components
    tp = sum(y_true == 1 & y_pred == 1);
    fp = sum(y_true == 0 & y_pred == 1);
    fn = sum(y_true == 1 & y_pred == 0);
    tn = sum(y_true == 0 & y_pred == 0);
    
    % Calculate metrics
    accuracy = (tp + tn) / (tp + fp + fn + tn);
    
    if (tp + fp) > 0
        precision = tp / (tp + fp);
    else
        precision = 0;
    end
    
    if (tp + fn) > 0
        recall = tp / (tp + fn);
    else
        recall = 0;
    end
    
    if (precision + recall) > 0
        f1 = 2 * (precision * recall) / (precision + recall);
    else
        f1 = 0;
    end
end

%% Helper function to rotate x-axis labels
function rotateXLabels(ax, angle)
    set(ax, 'XTickLabelRotation', angle);
end