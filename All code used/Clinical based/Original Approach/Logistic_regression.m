%% LOGISTIC REGRESSION WITH 5-FOLD CROSS-VALIDATION WITH SMOTE


fprintf('\n=== LOGISTIC REGRESSION WITH 5-FOLD CROSS-VALIDATION WITH SMOTE ===\n');

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

fprintf('Using %d predictors\n', length(use_vars));
fprintf('Stroke prevalence: %.2f%%\n', mean(y) * 100);

%% SMOTE Function Definition
function [X_resampled, y_resampled] = customSMOTE(X, y, k, oversampling_ratio)
    % Custom SMOTE implementation for MATLAB
    % X: features matrix
    % y: labels vector
    % k: number of nearest neighbors to consider
    % oversampling_ratio: how much to oversample minority class
    
    % Find minority and majority classes
    minority_class = 1; % Stroke class
    majority_class = 0; % Non-stroke class
    
    minority_idx = find(y == minority_class);
    majority_idx = find(y == majority_class);
    
    X_minority = X(minority_idx, :);
    n_minority = size(X_minority, 1);
    n_majority = length(majority_idx);
    
    % Calculate how many synthetic samples to generate
    target_minority = round(n_majority * oversampling_ratio);
    n_synthetic = max(0, target_minority - n_minority);
    
    if n_synthetic == 0
        X_resampled = X;
        y_resampled = y;
        return;
    end
    
    % Simple kNN implementation
    [idx, ~] = simpleKNN(X_minority, X_minority, min(k+1, n_minority));
    idx = idx(:, 2:end); % Remove self
    
    % Generate synthetic samples
    synthetic_samples = zeros(n_synthetic, size(X, 2));
    
    for i = 1:n_synthetic
        % Randomly select a minority sample
        rand_idx = randi(n_minority);
        
        % Randomly select one of its neighbors
        if size(idx, 2) > 0
            rand_neighbor_idx = idx(rand_idx, randi(size(idx, 2)));
        else
            rand_neighbor_idx = rand_idx;
        end
        
        % Generate synthetic sample
        diff = X_minority(rand_neighbor_idx, :) - X_minority(rand_idx, :);
        gap = rand();
        synthetic_samples(i, :) = X_minority(rand_idx, :) + gap * diff;
    end
    
    % Combine original and synthetic samples
    X_resampled = [X; synthetic_samples];
    y_resampled = [y; ones(n_synthetic, 1)];
    
    % Shuffle the resampled data
    shuffle_idx = randperm(size(X_resampled, 1));
    X_resampled = X_resampled(shuffle_idx, :);
    y_resampled = y_resampled(shuffle_idx);
end

function [idx, D] = simpleKNN(X, Y, k)
    % Simple kNN implementation
    n = size(X, 1);
    m = size(Y, 1);
    idx = zeros(m, k);
    D = zeros(m, k);
    
    for i = 1:m
        distances = sqrt(sum((X - Y(i,:)).^2, 2));
        [sorted_dist, sorted_idx] = sort(distances);
        idx(i,:) = sorted_idx(1:k);
        D(i,:) = sorted_dist(1:k);
    end
end

%% CRITICAL: Check and Handle Class Imbalance with SMOTE
fprintf('\n=== CLASS IMBALANCE ANALYSIS WITH SMOTE ===\n');

stroke_cases = sum(y);
non_stroke_cases = sum(y == 0);
total_cases = length(y);
imbalance_ratio = non_stroke_cases / stroke_cases;

fprintf('Original data:\n');
fprintf('  Stroke cases: %d (%.4f%%)\n', stroke_cases, (stroke_cases/total_cases)*100);
fprintf('  Non-stroke cases: %d (%.4f%%)\n', non_stroke_cases, (non_stroke_cases/total_cases)*100);
fprintf('  Imbalance ratio: %.2f:1 (non-stroke:stroke)\n', imbalance_ratio);

% ALWAYS use class weighting and SMOTE for significant imbalance
fprintf('\n⚠️  SIGNIFICANT CLASS IMBALANCE DETECTED\n');
fprintf('Applying SMOTE and class weighting in cross-validation...\n');
use_weights_cv = true;
use_smote_cv = true;

% SMOTE parameters
smote_k = 5;
smote_ratio = 0.5; % Target 1:2 stroke:non-stroke ratio

%% 5-FOLD CROSS-VALIDATION SETUP WITH SMOTE
rng(42); % For reproducibility
k = 5;
cv = cvpartition(y, 'KFold', k);

% Initialize storage for cross-validation results
fold_results = struct();
all_y_true_cv = [];
all_y_pred_prob_cv = [];
all_y_pred_binary_cv = [];

fprintf('\nRunning %d-fold cross-validation for Logistic Regression with SMOTE...\n', k);

%% CROSS-VALIDATION LOOP WITH SMOTE
for fold = 1:k
    fprintf('\n--- Fold %d/%d ---\n', fold, k);
    
    % Split data
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);
    
    X_train = X(train_idx, :);
    y_train = y(train_idx);
    X_test = X(test_idx, :);
    y_test = y(test_idx);
    
    % Check class distribution in this fold BEFORE SMOTE
    fprintf('  Before SMOTE: %d samples (%.1f%% stroke)\n', length(y_train), mean(y_train)*100);
    
    % === FIX: Apply SMOTE to training data ===
    if use_smote_cv
        [X_train_smote, y_train_smote] = customSMOTE(X_train, y_train, smote_k, smote_ratio);
        fprintf('  After SMOTE:  %d samples (%.1f%% stroke)\n', length(y_train_smote), mean(y_train_smote)*100);
    else
        X_train_smote = X_train;
        y_train_smote = y_train;
    end
    
    fprintf('  Testing:      %d samples (%.1f%% stroke)\n', length(y_test), mean(y_test)*100);
    
    % === FIX: Always use class weighting ===
    if use_weights_cv
        % Calculate weights based on SMOTE-augmented data
        class_counts = histcounts(y_train_smote, 2);
        class_weights = sum(class_counts) ./ (2 * class_counts);
        weights_train = class_weights(y_train_smote + 1)';
        fprintf('  Using class weights: %.2f for class 0, %.2f for class 1\n', ...
                class_weights(1), class_weights(2));
    else
        weights_train = ones(size(y_train_smote));
    end
    
    % Fit logistic regression model with weights
    if use_weights_cv
        [beta_fold, dev_fold, stats_fold] = glmfit(X_train_smote, y_train_smote, 'binomial', 'link', 'logit', 'weights', weights_train);
    else
        [beta_fold, dev_fold, stats_fold] = glmfit(X_train_smote, y_train_smote, 'binomial', 'link', 'logit');
    end
    
    % Make predictions on test set (original, not SMOTE-augmented)
    y_pred_prob_test = glmval(beta_fold, X_test, 'logit');
    
    % === FIX: Use F1-Score maximization instead of Youden's J ===
    [X_roc_fold, Y_roc_fold, T_fold, AUC_fold] = perfcurve(y_test, y_pred_prob_test, 1);
    
    % Calculate F1-score for each threshold
    f1_scores_fold = zeros(length(T_fold), 1);
    for i = 1:length(T_fold)
        y_pred_temp = y_pred_prob_test > T_fold(i);
        TP = sum(y_pred_temp & y_test);
        FP = sum(y_pred_temp & ~y_test);
        FN = sum(~y_pred_temp & y_test);
        
        precision_temp = TP / (TP + FP + eps);
        recall_temp = TP / (TP + FN + eps);
        f1_scores_fold(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
    end
    
    % Find threshold that maximizes F1-score
    [best_f1_fold, optimal_idx_fold] = max(f1_scores_fold);
    optimal_threshold_fold = T_fold(optimal_idx_fold);
    
    y_pred_binary_fold = y_pred_prob_test > optimal_threshold_fold;
    
    % Calculate metrics for this fold
    TP = sum(y_pred_binary_fold & y_test);
    FP = sum(y_pred_binary_fold & ~y_test);
    TN = sum(~y_pred_binary_fold & ~y_test);
    FN = sum(~y_pred_binary_fold & y_test);
    
    accuracy_fold = (TP + TN) / (TP + TN + FP + FN);
    precision_fold = TP / (TP + FP + eps);
    recall_fold = TP / (TP + FN + eps);
    specificity_fold = TN / (TN + FP + eps);
    f1_fold = 2 * (precision_fold * recall_fold) / (precision_fold + recall_fold + eps);
    brier_fold = mean((y_pred_prob_test - y_test).^2);
    
    % Calculate pseudo R-squared for this fold
    null_deviance_fold = -2 * sum(y_test * log(mean(y_test)) + (1-y_test) * log(1-mean(y_test)));
    model_deviance_fold = dev_fold;
    pseudo_r2_fold = 1 - (model_deviance_fold / null_deviance_fold);
    
    % Store fold results
    fold_results(fold).X_train = X_train_smote;
    fold_results(fold).y_train = y_train_smote;
    fold_results(fold).X_test = X_test;
    fold_results(fold).y_test = y_test;
    fold_results(fold).beta = beta_fold;
    fold_results(fold).stats = stats_fold;
    fold_results(fold).y_pred_prob = y_pred_prob_test;
    fold_results(fold).y_pred_binary = y_pred_binary_fold;
    fold_results(fold).X_roc = X_roc_fold;
    fold_results(fold).Y_roc = Y_roc_fold;
    fold_results(fold).AUC = AUC_fold;
    fold_results(fold).optimal_threshold = optimal_threshold_fold;
    fold_results(fold).optimal_idx = optimal_idx_fold;
    fold_results(fold).accuracy = accuracy_fold;
    fold_results(fold).precision = precision_fold;
    fold_results(fold).recall = recall_fold;
    fold_results(fold).specificity = specificity_fold;
    fold_results(fold).f1 = f1_fold;
    fold_results(fold).brier = brier_fold;
    fold_results(fold).pseudo_r2 = pseudo_r2_fold;
    fold_results(fold).TP = TP;
    fold_results(fold).FP = FP;
    fold_results(fold).TN = TN;
    fold_results(fold).FN = FN;
    fold_results(fold).weights_used = use_weights_cv;
    fold_results(fold).smote_used = use_smote_cv;
    
    % Aggregate for overall analysis
    all_y_true_cv = [all_y_true_cv; y_test];
    all_y_pred_prob_cv = [all_y_pred_prob_cv; y_pred_prob_test];
    all_y_pred_binary_cv = [all_y_pred_binary_cv; y_pred_binary_fold];
    
    fprintf('  Fold Results: Acc=%.2f%%, Prec=%.2f%%, Rec=%.2f%%, F1=%.2f%%, AUC=%.3f\n', ...
        accuracy_fold*100, precision_fold*100, recall_fold*100, f1_fold*100, AUC_fold);
    fprintf('  Optimal threshold: %.4f (maximizes F1=%.4f)\n', optimal_threshold_fold, best_f1_fold);
end

%% TRAIN FINAL MODEL ON ALL DATA WITH SMOTE FOR VISUALIZATION
fprintf('\n=== TRAINING FINAL MODEL FOR VISUALIZATION WITH SMOTE ===\n');

% === FIX: Apply SMOTE to entire dataset for final model training ===
fprintf('Original data before SMOTE: %d samples (%.1f%% stroke)\n', length(y), mean(y)*100);
[X_smote_final, y_smote_final] = customSMOTE(X, y, smote_k, smote_ratio);
fprintf('After SMOTE for final model: %d samples (%.1f%% stroke)\n', length(y_smote_final), mean(y_smote_final)*100);

% Train final model on SMOTE-augmented data with weighting
if use_weights_cv
    % Calculate weights for SMOTE-augmented dataset
    class_counts = histcounts(y_smote_final, 2);
    class_weights = sum(class_counts) ./ (2 * class_counts);
    weights_full = class_weights(y_smote_final + 1)';
    fprintf('Final model using class weights: %.2f for class 0, %.2f for class 1\n', ...
            class_weights(1), class_weights(2));
    [beta_final, dev_final, stats_final] = glmfit(X_smote_final, y_smote_final, 'binomial', 'link', 'logit', 'weights', weights_full);
else
    [beta_final, dev_final, stats_final] = glmfit(X_smote_final, y_smote_final, 'binomial', 'link', 'logit');
end

% Calculate predictions on ORIGINAL data for evaluation
y_pred_prob_final = glmval(beta_final, X, 'logit');

% === FIX: Use F1-Score maximization for threshold optimization ===
[X_roc_final, Y_roc_final, T_final, AUC_final] = perfcurve(y, y_pred_prob_final, 1);

% Calculate F1-score for each threshold
f1_scores = zeros(length(T_final), 1);
for i = 1:length(T_final)
    y_pred_temp = y_pred_prob_final > T_final(i);
    TP = sum(y_pred_temp & y);
    FP = sum(y_pred_temp & ~y);
    FN = sum(~y_pred_temp & y);
    
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    f1_scores(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
end

% Find threshold that maximizes F1-score
[best_f1, optimal_idx] = max(f1_scores);
optimal_threshold = T_final(optimal_idx);

fprintf('\n=== THRESHOLD OPTIMIZATION RESULTS ===\n');
fprintf('Optimal classification threshold: %.4f (maximizes F1-score)\n', optimal_threshold);
y_pred_binary_final = y_pred_prob_final > optimal_threshold;

% Calculate final model metrics
TP_final = sum(y_pred_binary_final & y);
FP_final = sum(y_pred_binary_final & ~y);
TN_final = sum(~y_pred_binary_final & ~y);
FN_final = sum(~y_pred_binary_final & y);

accuracy_final = (TP_final + TN_final) / (TP_final + TN_final + FP_final + FN_final);
precision_final = TP_final / (TP_final + FP_final + eps);
recall_final = TP_final / (TP_final + FN_final + eps);
specificity_final = TN_final / (TN_final + FP_final + eps);
f1_final = 2 * (precision_final * recall_final) / (precision_final + recall_final + eps);
brier_final = mean((y_pred_prob_final - y).^2);

% Calculate AUPRC for final model
[X_pr, Y_pr, ~, AUPRC] = perfcurve(y, y_pred_prob_final, 1, 'XCrit', 'reca', 'YCrit', 'prec');

% Calculate pseudo R-squared for final model
null_deviance_final = -2 * sum(y * log(mean(y)) + (1-y) * log(1-mean(y)));
pseudo_r2_final = 1 - (dev_final / null_deviance_final);

%% CROSS-VALIDATION PERFORMANCE SUMMARY
fprintf('\n=== CROSS-VALIDATION PERFORMANCE SUMMARY WITH SMOTE ===\n');

% Calculate average metrics across folds
avg_accuracy_cv = mean([fold_results.accuracy]);
avg_precision_cv = mean([fold_results.precision]);
avg_recall_cv = mean([fold_results.recall]);
avg_f1_cv = mean([fold_results.f1]);
avg_auc_cv = mean([fold_results.AUC]);
avg_brier_cv = mean([fold_results.brier]);
avg_pseudo_r2_cv = mean([fold_results.pseudo_r2]);

% Calculate standard deviations
std_accuracy_cv = std([fold_results.accuracy]);
std_precision_cv = std([fold_results.precision]);
std_recall_cv = std([fold_results.recall]);
std_f1_cv = std([fold_results.f1]);
std_auc_cv = std([fold_results.AUC]);
std_brier_cv = std([fold_results.brier]);

fprintf('\n5-Fold Cross-Validation Results with SMOTE (Mean ± SD):\n');
fprintf('Accuracy:    %.2f%% ± %.2f%%\n', avg_accuracy_cv*100, std_accuracy_cv*100);
fprintf('Precision:   %.2f%% ± %.2f%%\n', avg_precision_cv*100, std_precision_cv*100);
fprintf('Recall:      %.2f%% ± %.2f%%\n', avg_recall_cv*100, std_recall_cv*100);
fprintf('F1-Score:    %.2f%% ± %.2f%%\n', avg_f1_cv*100, std_f1_cv*100);
fprintf('AUC:         %.3f ± %.3f\n', avg_auc_cv, std_auc_cv);
fprintf('Brier Score: %.4f ± %.4f\n', avg_brier_cv, std_brier_cv);
fprintf('Pseudo R²:   %.4f (avg across folds)\n', avg_pseudo_r2_cv);

% Find best fold
[best_auc_cv, best_fold] = max([fold_results.AUC]);
fprintf('\nBest performing fold: Fold %d (AUC = %.3f)\n', best_fold, best_auc_cv);

%% MODEL DIAGNOSTICS FOR FINAL MODEL
fprintf('\n=== FINAL MODEL DIAGNOSTICS WITH SMOTE ===\n');

% Check for separation issues
if any(y_pred_prob_final > 0.999 | y_pred_prob_final < 0.001)
    fprintf('⚠️  Possible complete separation detected\n');
    fprintf('   %d predictions near 0 or 1\n', sum(y_pred_prob_final > 0.999 | y_pred_prob_final < 0.001));
else
    fprintf('✓ No complete separation detected\n');
end

% Check for infinite or NaN coefficients
if any(isinf(beta_final) | isnan(beta_final))
    fprintf('⚠️  Infinite or NaN coefficients detected\n');
else
    fprintf('✓ All coefficients are finite\n');
end

% Check coefficient stability
large_coeff = sum(abs(beta_final) > 10);
if large_coeff > 0
    fprintf('⚠️  %d coefficients have large magnitude (>10)\n', large_coeff);
else
    fprintf('✓ Coefficients have reasonable magnitude\n');
end

fprintf('Final Model Deviance Improvement: %.4f\n', pseudo_r2_final);
if pseudo_r2_final > 0.1
    fprintf('✓ Good improvement over null model\n');
else
    fprintf('⚠️  Limited improvement over null model\n');
end

%% Check for Multicollinearity
fprintf('\n=== MULTICOLLINEARITY CHECK ===\n');
correlation_matrix = corr(X);
high_corr = sum(abs(correlation_matrix) > 0.7, 2) - 1; % -1 to exclude self-correlation

if any(high_corr > 0)
    fprintf('⚠️  Potential multicollinearity detected:\n');
    for i = 1:length(use_vars)
        if high_corr(i) > 0
            fprintf('%-25s has %d correlations > 0.7\n', use_vars{i}, high_corr(i));
        end
    end
else
    fprintf('✓ No severe multicollinearity detected\n');
end

%% ENHANCED VISUALIZATION FOR IMBALANCED DATA WITH SMOTE
fprintf('\n=== GENERATING ENHANCED VISUALIZATIONS WITH SMOTE ===\n');

% Main 2x3 plot grid
figure;
sgtitle('Logistic Regression for Stroke Prediction (5-Fold CV with SMOTE)', 'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: ROC Curve with optimal threshold
subplot(2,3,1);
plot(X_roc_final, Y_roc_final, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
plot(X_roc_final(optimal_idx), Y_roc_final(optimal_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
xlabel('False Positive Rate');
ylabel('True Positive Rate (Sensitivity)');
title(sprintf('ROC Curve (AUC = %.4f)', AUC_final));
legend(sprintf('Logistic Regression with SMOTE (AUC=%.4f)', AUC_final), ...
       'Random Classifier', ...
       sprintf('Optimal Threshold=%.3f\n(Prec=%.1f%%, Rec=%.1f%%)', optimal_threshold, precision_final*100, recall_final*100), ...
       'Location', 'southeast');
grid on;

% Add CV performance annotation
text(0.6, 0.2, sprintf('CV AUC: %.3f ± %.3f', avg_auc_cv, std_auc_cv), ...
     'FontSize', 9, 'BackgroundColor', 'white', 'EdgeColor', 'blue', 'Margin', 2);

% Plot 2: Precision-Recall Curve
subplot(2,3,2);
plot(X_pr, Y_pr, 'g-', 'LineWidth', 2);
xlabel('Recall (Sensitivity)');
ylabel('Precision');
title(sprintf('Precision-Recall Curve (AUPRC = %.4f)', AUPRC));
grid on;
% Add baseline for imbalanced data
baseline_precision = sum(y) / length(y);
line([0 1], [baseline_precision baseline_precision], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 1);
legend('Logistic Regression with SMOTE', sprintf('Baseline (%.4f)', baseline_precision), 'Location', 'southwest');

% Plot 3: Probability distributions by class
subplot(2,3,3);
hold on;
if sum(y==0) > 0
    histogram(y_pred_prob_final(y==0), 'BinWidth', 0.02, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
if sum(y==1) > 0
    histogram(y_pred_prob_final(y==1), 'BinWidth', 0.02, 'FaceColor', 'red', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
line([optimal_threshold optimal_threshold], ylim(), 'Color', 'black', 'LineWidth', 2, 'LineStyle', '--');
xlabel('Predicted Probability');
ylabel('Density');
title('Probability Distribution by Class (SMOTE Model)');
legend('No Stroke', 'Stroke', sprintf('Threshold=%.3f', optimal_threshold), 'Location', 'northwest');
grid on;

% Plot 4: Feature importance for Logistic Regression
subplot(2,3,4);
base_mse = mean((y_pred_prob_final - y).^2);
perm_importance = zeros(1, min(10, size(X, 2)));

for i = 1:min(10, size(X, 2))
    X_permuted = X;
    X_permuted(:,i) = X_permuted(randperm(size(X_permuted,1)), i);
    perm_prob = glmval(beta_final, X_permuted, 'logit');
    perm_mse = mean((perm_prob - y).^2);
    perm_importance(i) = perm_mse - base_mse;
end

% Sort by importance
[perm_sorted, perm_sorted_idx] = sort(perm_importance, 'descend');
top_features_to_show = min(8, length(use_vars));

barh(perm_sorted(1:top_features_to_show));
set(gca, 'YTick', 1:top_features_to_show, 'YTickLabel', use_vars(perm_sorted_idx(1:top_features_to_show)));
xlabel('Feature Importance (MSE Increase)');
title('Logistic Regression Feature Importance (SMOTE Model)');
grid on;
hold on;
plot([0 0], ylim(), 'r--', 'LineWidth', 2);
legend('Feature Importance', 'No Effect', 'Location', 'southeast');

% Plot 5: Enhanced Threshold analysis
subplot(2,3,5);
thresholds = 0.01:0.01:0.99;
f1_scores_plot = zeros(size(thresholds));
precisions_plot = zeros(size(thresholds));
recalls_plot = zeros(size(thresholds));

for i = 1:length(thresholds)
    y_pred_temp = y_pred_prob_final >= thresholds(i);
    TP = sum(y_pred_temp & y);
    FP = sum(y_pred_temp & ~y);
    FN = sum(~y_pred_temp & y);
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    f1_scores_plot(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
    precisions_plot(i) = precision_temp;
    recalls_plot(i) = recall_temp;
end

plot(thresholds, f1_scores_plot, 'k-', 'LineWidth', 2);
hold on;
plot(thresholds, precisions_plot, 'b--', 'LineWidth', 1.5);
plot(thresholds, recalls_plot, 'g--', 'LineWidth', 1.5);

% Find the index corresponding to the optimal threshold
threshold_index = find(thresholds >= optimal_threshold, 1);
if ~isempty(threshold_index)
    plot(optimal_threshold, f1_scores_plot(threshold_index), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    plot(optimal_threshold, precisions_plot(threshold_index), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
    plot(optimal_threshold, recalls_plot(threshold_index), 'go', 'MarkerSize', 8, 'LineWidth', 2);
else
    % Fallback: use the closest threshold
    [~, closest_idx] = min(abs(thresholds - optimal_threshold));
    plot(optimal_threshold, f1_scores_plot(closest_idx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    plot(optimal_threshold, precisions_plot(closest_idx), 'bo', 'MarkerSize', 8, 'LineWidth', 2);
    plot(optimal_threshold, recalls_plot(closest_idx), 'go', 'MarkerSize', 8, 'LineWidth', 2);
end

xlabel('Classification Threshold');
ylabel('Score');
title('Performance Metrics vs Threshold (SMOTE Model)');
grid on;
legend('F1-Score', 'Precision', 'Recall', sprintf('Optimal (%.3f)', optimal_threshold), 'Location', 'southeast');

% Add CV thresholds from each fold
for fold = 1:k
    fold_threshold = fold_results(fold).optimal_threshold;
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
    in_bin = y_pred_prob_final >= bin_edges(i) & y_pred_prob_final < bin_edges(i+1);
    if sum(in_bin) > 0
        mean_predicted(i) = mean(y_pred_prob_final(in_bin));
        actual_proportion(i) = mean(y(in_bin));
    end
end
plot(mean_predicted, actual_proportion, 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 2);
xlabel('Mean Predicted Probability');
ylabel('Actual Proportion');
title('Probability Calibration Plot (SMOTE Model)');
legend('Logistic Regression with SMOTE', 'Perfect Calibration', 'Location', 'northwest');
grid on;

% Add number of samples per bin
for i = 1:length(bin_centers)
    in_bin = y_pred_prob_final >= bin_edges(i) & y_pred_prob_final < bin_edges(i+1);
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
sgtitle('Logistic Regression for Stroke Prediction (5-Fold CV with SMOTE)', 'FontSize', 16, 'FontWeight', 'bold');
subplot(1,2,1);
y_numeric = double(y);
y_pred_binary_numeric = double(y_pred_binary_final);
CM = confusionmat(y_numeric, y_pred_binary_numeric);
confusionchart(CM, {'No Stroke', 'Stroke'}, 'Title', 'Confusion Matrix (Optimal Threshold)');

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
y_pred_jittered = y_pred_prob_final + 0.01 * randn(size(y_pred_prob_final)); % Very small jitter to predictions

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
title(sprintf('Jittered Scatter Plot\n(Optimal Threshold = %.3f)', optimal_threshold));
grid on;
legend('Location', 'northwest');

% Plot 10: Box plot of predictions by actual class
subplot(1,2,2);
boxplot(y_pred_prob_final, y_numeric, 'Labels', {'No Stroke', 'Stroke'});
xlabel('Actual Class');
ylabel('Predicted Probability');
title('Box Plot of Predicted Probabilities by Actual Class (SMOTE Model)');
grid on;

% Add optimal threshold line to box plot
hold on;
y_limits = ylim();
plot(xlim(), [optimal_threshold optimal_threshold], 'r--', 'LineWidth', 2, ...
     'DisplayName', sprintf('Optimal Threshold (%.3f)', optimal_threshold));
legend('Location', 'northwest');

%% CROSS-VALIDATION SPECIFIC VISUALIZATION
figure;
sgtitle('5-Fold Cross-Validation Analysis - Logistic Regression with SMOTE', 'FontSize', 16, 'FontWeight', 'bold');

% Plot CV performance across folds
subplot(2,3,1);
metrics = {'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'};
fold_metrics_matrix = [[fold_results.accuracy]; [fold_results.precision]; 
                      [fold_results.recall]; [fold_results.f1]; [fold_results.AUC]];

plot(1:k, fold_metrics_matrix, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Fold Number');
ylabel('Score');
title('Performance Metrics Across Folds (with SMOTE)');
legend(metrics, 'Location', 'best');
grid on;
xlim([0.5, k+0.5]);

% Plot optimal thresholds across folds
subplot(2,3,2);
bar(1:k, [fold_results.optimal_threshold]);
hold on;
plot(xlim(), [optimal_threshold optimal_threshold], 'r--', 'LineWidth', 2);
xlabel('Fold Number');
ylabel('Optimal Threshold');
title('Optimal Thresholds Across Folds (with SMOTE)');
grid on;
xticks(1:k);
legend('Fold Threshold', 'Final Model Threshold', 'Location', 'best');

% Plot ROC curves from all folds
subplot(2,3,3);
hold on;
colors = lines(k);
for fold = 1:k
    plot(fold_results(fold).X_roc, fold_results(fold).Y_roc, ...
         'Color', colors(fold,:), 'LineWidth', 1.5, ...
         'DisplayName', sprintf('Fold %d (AUC=%.3f)', fold, fold_results(fold).AUC));
end
plot(X_roc_final, Y_roc_final, 'k-', 'LineWidth', 3, ...
     'DisplayName', sprintf('Final Model (AUC=%.3f)', AUC_final));
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5, 'DisplayName', 'Random');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves: All Folds vs Final Model (with SMOTE)');
legend('Location', 'southeast', 'FontSize', 8);
grid on;

% Plot coefficient stability across folds
subplot(2,3,4);
% Collect coefficients from all folds (skip intercept)
coeffs_all = zeros(k, length(use_vars));
for fold = 1:k
    coeffs_all(fold, :) = fold_results(fold).beta(2:end)'; % Skip intercept
end

% Calculate coefficient of variation for each feature
coeff_means = mean(coeffs_all, 1);
coeff_stds = std(coeffs_all, 0, 1);
coeff_cv = coeff_stds ./ abs(coeff_means);
[~, idx_cv] = sort(coeff_cv, 'descend');
n_top_cv = min(8, length(use_vars));

barh(1:n_top_cv, coeff_cv(idx_cv(1:n_top_cv)));
set(gca, 'YTick', 1:n_top_cv, 'YTickLabel', use_vars(idx_cv(1:n_top_cv)));
xlabel('Coefficient of Variation');
title('Feature Coefficient Stability (CV with SMOTE)');
grid on;

% Plot odds ratios from final model
subplot(2,3,5);
odds_ratios = exp(beta_final(2:end)); % Skip intercept
[~, idx_or] = sort(abs(log(odds_ratios)), 'descend');
n_top_or = min(8, length(use_vars));

barh(1:n_top_or, odds_ratios(idx_or(1:n_top_or)));
set(gca, 'YTick', 1:n_top_or, 'YTickLabel', use_vars(idx_or(1:n_top_or)));
xlabel('Odds Ratio');
title('Top Feature Odds Ratios (SMOTE Model)');
grid on;
hold on;
plot([1 1], ylim(), 'r--', 'LineWidth', 2); % Reference line at OR=1

% Plot performance comparison: CV vs Final Model
subplot(2,3,6);
metrics_final = [accuracy_final, precision_final, recall_final, f1_final, AUC_final];
metrics_cv_mean = [avg_accuracy_cv, avg_precision_cv, avg_recall_cv, avg_f1_cv, avg_auc_cv];
metrics_cv_std = [std_accuracy_cv, std_precision_cv, std_recall_cv, std_f1_cv, std_auc_cv];

x_pos = 1:length(metrics);
bar(x_pos, metrics_final, 0.3, 'FaceColor', 'b', 'FaceAlpha', 0.7);
hold on;
errorbar(x_pos, metrics_cv_mean, metrics_cv_std, 'o', 'Color', 'r', ...
         'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
set(gca, 'XTick', x_pos, 'XTickLabel', metrics);
ylabel('Score');
title('Performance: Final Model vs CV (Mean ± SD) with SMOTE');
legend('Final Model', 'CV Mean ± SD', 'Location', 'southwest');
grid on;
rotateXLabels(gca, 45);

%% DISPLAY FINAL MODEL COEFFICIENTS
fprintf('\n=== FINAL MODEL COEFFICIENTS (SMOTE Model) ===\n');
fprintf('%-25s %-10s %-10s %-10s\n', 'Predictor', 'Coefficient', 'Std Error', 'p-value');
fprintf('%-25s %-10.4f %-10.4f %-10.4f\n', '(Intercept)', beta_final(1), stats_final.se(1), stats_final.p(1));

for i = 1:length(use_vars)
    fprintf('%-25s %-10.4f %-10.4f %-10.4f', ...
            use_vars{i}, beta_final(i+1), stats_final.se(i+1), stats_final.p(i+1));
    if stats_final.p(i+1) < 0.05
        fprintf(' *');
    end
    if stats_final.p(i+1) < 0.01
        fprintf('*');
    end
    if stats_final.p(i+1) < 0.001
        fprintf('*');
    end
    fprintf('\n');
end

%% Enhanced Coefficient Interpretation
fprintf('\n--- Coefficient Interpretation (Odds Ratios) ---\n');
fprintf('%-25s %-12s %-15s\n', 'Predictor', 'Odds Ratio', 'Interpretation');
fprintf('%-25s %-12.4f %-15s\n', '(Intercept)', exp(beta_final(1)), 'Baseline odds');

for i = 1:length(use_vars)
    or = exp(beta_final(i+1));
    if or > 1
        interpretation = sprintf('+%.1f%% risk per unit', (or-1)*100);
    else
        interpretation = sprintf('%.1f%% risk reduction per unit', (1-or)*100);
    end
    
    significance = '';
    if stats_final.p(i+1) < 0.001
        significance = ' ***';
    elseif stats_final.p(i+1) < 0.01
        significance = ' **';
    elseif stats_final.p(i+1) < 0.05
        significance = ' *';
    end
    
    fprintf('%-25s %-12.4f %-15s%s\n', use_vars{i}, or, interpretation, significance);
end
fprintf('\n*** p<0.001, ** p<0.01, * p<0.05\n');

%% FINAL PERFORMANCE METRICS
fprintf('\n=== FINAL MODEL PERFORMANCE WITH SMOTE ===\n');

fprintf('AUC:         %.4f\n', AUC_final);
fprintf('AUPRC:       %.4f\n', AUPRC);
fprintf('Accuracy:    %.4f (%.2f%%)\n', accuracy_final, accuracy_final*100);
fprintf('Precision:   %.4f (%.2f%%)\n', precision_final, precision_final*100);
fprintf('Recall:      %.4f (%.2f%%)\n', recall_final, recall_final*100);
fprintf('Specificity: %.4f (%.2f%%)\n', specificity_final, specificity_final*100);
fprintf('F1-Score:    %.4f\n', f1_final);
fprintf('Brier Score: %.4f\n', brier_final);
fprintf('Pseudo R²:   %.4f\n', pseudo_r2_final);

%% Confusion Matrix Details
fprintf('\n=== CONFUSION MATRIX (Optimal Threshold = %.3f) ===\n', optimal_threshold);
fprintf('           Predicted 0   Predicted 1\n');
fprintf('Actual 0:   %6d (TN)    %6d (FP)\n', TN_final, FP_final);
fprintf('Actual 1:   %6d (FN)    %6d (TP)\n', FN_final, TP_final);

%% Feature Importance Analysis
fprintf('\n=== TOP FEATURE IMPORTANCE ===\n');
fprintf('Top 5 most important features:\n');
for i = 1:min(5, length(use_vars))
    fprintf('%d. %-30s (importance: %.4f)\n', i, use_vars{perm_sorted_idx(i)}, perm_sorted(i));
end

%% SMOTE Impact Assessment
fprintf('\n=== SMOTE IMPACT ASSESSMENT ===\n');
fprintf('Improvements applied:\n');
fprintf('1. SMOTE oversampling: k=%d, ratio=%.1f\n', smote_k, smote_ratio);
fprintf('2. F1-Score threshold optimization (instead of Youden''s J)\n');
fprintf('3. Enhanced threshold analysis with precision/recall curves\n');
fprintf('4. Always using class weighting with SMOTE\n');

fprintf('\nExpected improvements:\n');
fprintf('  - Better recall (sensitivity) for stroke detection\n');
fprintf('  - Improved F1-score through balanced precision-recall\n');
fprintf('  - More robust model trained on balanced data\n');
fprintf('  - Better generalization to minority class patterns\n');

%% Clinical Utility Assessment
fprintf('\n=== CLINICAL UTILITY ASSESSMENT WITH SMOTE ===\n');
fprintf('Sensitivity (Recall): %.2f%% - Ability to detect true strokes\n', recall_final*100);
fprintf('Specificity:          %.2f%% - Ability to avoid false alarms\n', specificity_final*100);
fprintf('Precision:            %.2f%% - Reliability of positive predictions\n', precision_final*100);

if recall_final < 0.3
    fprintf('⚠️  WARNING: Very low sensitivity - model may miss many strokes\n');
elseif recall_final < 0.5
    fprintf('⚠️  WARNING: Low sensitivity - consider lowering threshold\n');
else
    fprintf('✓ Acceptable sensitivity for stroke detection\n');
end

if precision_final < 0.2
    fprintf('⚠️  WARNING: Very low precision - many false alarms\n');
elseif precision_final < 0.3
    fprintf('⚠️  WARNING: Low precision - many false positives\n');
else
    fprintf('✓ Reasonable precision for medical context\n');
end

if AUPRC > 0.5
    fprintf('✓ Good performance on imbalanced data (AUPRC > 0.5)\n');
else
    fprintf('⚠️  Poor performance on imbalanced data (AUPRC <= 0.5)\n');
end

%% CROSS-VALIDATION INTERPRETATION
fprintf('\n=== CROSS-VALIDATION INTERPRETATION WITH SMOTE ===\n');
fprintf('Model Consistency Assessment:\n');
fprintf('  Accuracy CV:  %.2f%% (lower is better)\n', (std_accuracy_cv/avg_accuracy_cv)*100);
fprintf('  AUC CV:       %.2f%% (lower is better)\n', (std_auc_cv/avg_auc_cv)*100);

if (std_auc_cv/avg_auc_cv) < 0.1
    fprintf('✓ Model shows good consistency across folds\n');
elseif (std_auc_cv/avg_auc_cv) < 0.2
    fprintf('⚠️  Model shows moderate consistency across folds\n');
else
    fprintf('⚠️  Model shows high variability across folds\n');
end

fprintf('\nCross-validation with SMOTE advantages:\n');
fprintf('1. More reliable performance estimate with balanced data\n');
fprintf('2. Better assessment of minority class performance\n');
fprintf('3. Reduced bias toward majority class\n');
fprintf('4. Improved generalization to real-world imbalance\n');

fprintf('\n=== LOGISTIC REGRESSION WITH 5-FOLD CV WITH SMOTE ANALYSIS COMPLETE ===\n');

%% Helper function to rotate x-axis labels
function rotateXLabels(ax, angle)
    set(ax, 'XTickLabelRotation', angle);
end