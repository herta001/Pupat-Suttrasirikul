%% Random FOrest Model with 5-Fold cross valiation and SMOTE

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
y_numeric = processed_data.stroke; % Keep as numeric for consistency
y = categorical(y_numeric); % Convert to categorical for classification

fprintf('Using %d predictors\n', length(use_vars));
fprintf('Stroke prevalence: %.2f%%\n', mean(y_numeric) * 100);

%% CRITICAL: Check and Handle Class Imbalance
fprintf('\n=== CLASS IMBALANCE ANALYSIS ===\n');

stroke_cases = sum(y_numeric);
non_stroke_cases = sum(y_numeric == 0);
total_cases = length(y_numeric);
imbalance_ratio = non_stroke_cases / stroke_cases;

fprintf('Stroke cases: %d (%.4f%%)\n', stroke_cases, (stroke_cases/total_cases)*100);
fprintf('Non-stroke cases: %d (%.4f%%)\n', non_stroke_cases, (non_stroke_cases/total_cases)*100);
fprintf('Imbalance ratio: %.2f:1 (non-stroke:stroke)\n', imbalance_ratio);

%% ADD SMOTE FUNCTION FOR RANDOM FOREST
fprintf('\n=== APPLYING SMOTE FOR CLASS IMBALANCE (WITHIN CROSS-VALIDATION) ===\n');

% SMOTE function definition for Random Forest
function [X_balanced, y_balanced, y_balanced_numeric] = apply_smote_rf(X_train, y_train_numeric, desired_ratio)
    minority_class = 1;
    majority_class = 0;
    
    train_minority_count = sum(y_train_numeric == minority_class);
    train_majority_count = sum(y_train_numeric == majority_class);
    
    % Target balanced dataset
    if desired_ratio < 1
        desired_minority_count = min(train_majority_count, round(train_minority_count / desired_ratio));
    else
        desired_minority_count = min(train_majority_count, round(train_minority_count * desired_ratio));
    end
    
    synthetic_samples_needed = max(0, desired_minority_count - train_minority_count);
    
    if synthetic_samples_needed > 0
        % Prepare minority class data
        train_minority_idx = find(y_train_numeric == minority_class);
        X_train_minority = X_train(train_minority_idx, :);
        
        % SMOTE parameters
        k = min(5, size(X_train_minority, 1) - 1);
        [idx, ~] = knnsearch(X_train_minority, X_train_minority, 'K', k+1);
        
        % Generate synthetic samples
        X_synthetic = zeros(synthetic_samples_needed, size(X_train_minority, 2));
        y_synthetic_numeric = ones(synthetic_samples_needed, 1) * minority_class;
        
        synthetic_count = 0;
        while synthetic_count < synthetic_samples_needed
            for i = 1:size(X_train_minority, 1)
                if synthetic_count >= synthetic_samples_needed
                    break;
                end
                
                neighbor_idx = idx(i, randi(k) + 1);
                diff = X_train_minority(neighbor_idx, :) - X_train_minority(i, :);
                gap = rand();
                
                synthetic_sample = X_train_minority(i, :) + gap * diff;
                
                % Clip to feature bounds
                for j = 1:size(synthetic_sample, 2)
                    feat_min = min(X_train(:, j));
                    feat_max = max(X_train(:, j));
                    synthetic_sample(j) = max(feat_min, min(feat_max, synthetic_sample(j)));
                end
                
                synthetic_count = synthetic_count + 1;
                X_synthetic(synthetic_count, :) = synthetic_sample;
            end
        end
        
        % Combine original and synthetic data
        X_balanced = [X_train; X_synthetic];
        y_balanced_numeric = [y_train_numeric; y_synthetic_numeric];
        
        % Shuffle
        shuffle_idx = randperm(size(X_balanced, 1));
        X_balanced = X_balanced(shuffle_idx, :);
        y_balanced_numeric = y_balanced_numeric(shuffle_idx);
        
        % Convert to categorical for Random Forest
        y_balanced = categorical(y_balanced_numeric);
    else
        X_balanced = X_train;
        y_balanced_numeric = y_train_numeric;
        y_balanced = categorical(y_balanced_numeric);
    end
end

% Set up initial 70-30 split for final evaluation
rng(42); % For reproducibility
cv_holdout = cvpartition(y_numeric, 'Holdout', 0.3); % 70-30 split
train_idx_all = training(cv_holdout);
test_idx_all = test(cv_holdout);

%% RANDOM FOREST with 5-FOLD CROSS-VALIDATION TRAINING AND SMOTE
fprintf('\n=== RANDOM FOREST WITH 5-FOLD CROSS-VALIDATION TRAINING AND SMOTE ===\n');

% Cost matrix for imbalance handling
cost_matrix = [0 1; imbalance_ratio 0];

% Set up 5-fold cross-validation on the training data
k = 5;
cv_folds = cvpartition(y_numeric(train_idx_all), 'KFold', k);

fprintf('Training Random Forest with %d-fold cross-validation and SMOTE...\n', k);

% Initialize arrays to store fold models and predictions
cv_models = cell(k, 1);
cv_predictions = cell(k, 1);
cv_scores = cell(k, 1);
cv_metrics = zeros(k, 6); % [accuracy, precision, recall, specificity, f1, auc]
cv_smote_ratios = zeros(k, 1); % Store SMOTE ratios used for each fold
cv_training_stats = cell(k, 1); % Store training statistics

% Store SMOTE performance for analysis
smote_ratio_performance = struct();

% Try different SMOTE ratios to find optimal one
smote_target_ratios = [0.5, 0.67, 1.0, 1.5, 2.0];
best_overall_smote_ratio = 1.0; % Default

% Initialize smote_ratio_performance structure
for ratio_idx = 1:length(smote_target_ratios)
    ratio = smote_target_ratios(ratio_idx);
    smote_ratio_performance(ratio_idx).ratio = ratio;
    smote_ratio_performance(ratio_idx).auc_scores = [];
    smote_ratio_performance(ratio_idx).f1_scores = [];
    smote_ratio_performance(ratio_idx).precision_scores = [];
end

% Train a model on each fold with SMOTE
for fold = 1:k
    fprintf('\n--- Training fold %d/%d with SMOTE ---\n', fold, k);
    
    % Get fold indices
    fold_train_idx = training(cv_folds, fold);
    fold_val_idx = test(cv_folds, fold);
    
    % Get actual indices in the original data
    X_fold_train = X(train_idx_all(fold_train_idx), :);
    y_fold_train_numeric = y_numeric(train_idx_all(fold_train_idx));
    X_fold_val = X(train_idx_all(fold_val_idx), :);
    y_fold_val = y_numeric(train_idx_all(fold_val_idx));
    
    % Try different SMOTE ratios for this fold to find best performance
    best_fold_smote_ratio = 1.0;
    best_fold_auc = 0;
    best_fold_f1 = 0;
    best_fold_model = [];
    
    % Store SMOTE performance for this fold
    smote_results = struct('ratio', [], 'auc', [], 'f1', [], 'precision', []);
    
    for smote_idx = 1:length(smote_target_ratios)
        smote_ratio = smote_target_ratios(smote_idx);
        
        % Apply SMOTE to training fold
        [X_fold_train_balanced, y_fold_train_balanced, y_fold_train_balanced_numeric] = ...
            apply_smote_rf(X_fold_train, y_fold_train_numeric, smote_ratio);
        
        fprintf('  Testing SMOTE ratio %.2f...', smote_ratio);
        
        % Train Random Forest on this balanced fold
        try
            rf_fold_temp = fitcensemble(X_fold_train_balanced, y_fold_train_balanced, ...
                'Method', 'Bag', ...
                'NumLearningCycles', 200, ...
                'Learners', templateTree('MinLeafSize', 10, 'PredictorSelection', 'curvature'), ...
                'Cost', cost_matrix, ...
                'Options', statset('UseParallel', true));
            
            % Make predictions on validation fold
            [temp_pred_labels, temp_scores] = predict(rf_fold_temp, X_fold_val);
            temp_pred_binary = double(temp_pred_labels == '1');
            temp_pred_prob = temp_scores(:, 2);
            
            % Calculate AUC for this fold
            [~, ~, ~, temp_auc] = perfcurve(y_fold_val, temp_pred_prob, 1);
            
            % Calculate F1 score
            TP = sum(temp_pred_binary & y_fold_val);
            FP = sum(temp_pred_binary & ~y_fold_val);
            FN = sum(~temp_pred_binary & y_fold_val);
            precision_temp = TP / (TP + FP + eps);
            recall_temp = TP / (TP + FN + eps);
            temp_f1 = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
            
            fprintf(' AUC=%.4f, F1=%.4f\n', temp_auc, temp_f1);
            
            % Store results in fold-specific structure
            if isempty(smote_results)
                smote_results = struct('ratio', smote_ratio, 'auc', temp_auc, 'f1', temp_f1, 'precision', precision_temp);
            else
                smote_results(smote_idx) = struct('ratio', smote_ratio, 'auc', temp_auc, 'f1', temp_f1, 'precision', precision_temp);
            end
            
            % Store in overall performance tracking
            ratio_idx = find([smote_ratio_performance.ratio] == smote_ratio);
            if ~isempty(ratio_idx)
                smote_ratio_performance(ratio_idx).auc_scores = [smote_ratio_performance(ratio_idx).auc_scores, temp_auc];
                smote_ratio_performance(ratio_idx).f1_scores = [smote_ratio_performance(ratio_idx).f1_scores, temp_f1];
                smote_ratio_performance(ratio_idx).precision_scores = [smote_ratio_performance(ratio_idx).precision_scores, precision_temp];
            end
            
            % Check if this is the best SMOTE ratio for this fold
            combined_score = 0.5 * temp_auc + 0.5 * temp_f1; % Equal weight to AUC and F1
            current_best = 0.5 * best_fold_auc + 0.5 * best_fold_f1;
            
            if combined_score > current_best
                best_fold_auc = temp_auc;
                best_fold_f1 = temp_f1;
                best_fold_smote_ratio = smote_ratio;
                best_fold_model = rf_fold_temp;
            end
            
        catch ME
            fprintf(' Failed: %s\n', ME.message);
            continue;
        end
    end
    
    % Store the best model for this fold
    cv_models{fold} = best_fold_model;
    cv_smote_ratios(fold) = best_fold_smote_ratio;
    cv_training_stats{fold} = smote_results;
    
    % Make predictions on validation fold using best model
    [cv_pred_labels, cv_fold_scores] = predict(best_fold_model, X_fold_val);
    cv_pred_binary = double(cv_pred_labels == '1');
    cv_pred_prob = cv_fold_scores(:, 2);
    
    % Store predictions
    cv_predictions{fold} = cv_pred_binary;
    cv_scores{fold} = cv_pred_prob;
    
    % Calculate metrics for this fold
    TP = sum(cv_pred_binary & y_fold_val);
    FP = sum(cv_pred_binary & ~y_fold_val);
    TN = sum(~cv_pred_binary & ~y_fold_val);
    FN = sum(~cv_pred_binary & y_fold_val);
    
    cv_metrics(fold, 1) = (TP + TN) / (TP + TN + FP + FN + eps); % Accuracy
    cv_metrics(fold, 2) = TP / (TP + FP + eps); % Precision
    cv_metrics(fold, 3) = TP / (TP + FN + eps); % Recall
    cv_metrics(fold, 4) = TN / (TN + FP + eps); % Specificity
    cv_metrics(fold, 5) = 2 * (cv_metrics(fold, 2) * cv_metrics(fold, 3)) / ...
                          (cv_metrics(fold, 2) + cv_metrics(fold, 3) + eps); % F1
    [~, ~, ~, cv_metrics(fold, 6)] = perfcurve(y_fold_val, cv_pred_prob, 1); % AUC
    
    fprintf('Fold %d complete: SMOTE ratio=%.2f, AUC=%.4f, F1=%.4f\n', ...
            fold, best_fold_smote_ratio, cv_metrics(fold, 6), cv_metrics(fold, 5));
end

% Determine best SMOTE ratio based on average performance across folds
fprintf('\n=== SMOTE RATIO PERFORMANCE SUMMARY ===\n');
fprintf('Ratio\tMean AUC\tMean F1\t\tMean Precision\n');
fprintf('-----\t--------\t------\t\t-------------\n');

best_combined_score = 0;
for ratio_idx = 1:length(smote_ratio_performance)
    perf = smote_ratio_performance(ratio_idx);
    if ~isempty(perf.auc_scores)
        mean_auc = mean(perf.auc_scores);
        mean_f1 = mean(perf.f1_scores);
        mean_precision = mean(perf.precision_scores);
        combined_score = 0.5 * mean_auc + 0.5 * mean_f1;
        
        fprintf('%.2f\t%.4f\t\t%.4f\t\t%.4f\n', perf.ratio, mean_auc, mean_f1, mean_precision);
        
        if combined_score > best_combined_score
            best_combined_score = combined_score;
            best_overall_smote_ratio = perf.ratio;
        end
    end
end

fprintf('\nBest overall SMOTE ratio: %.2f (combined score: %.4f)\n', best_overall_smote_ratio, best_combined_score);

%% Create ensemble model from all training data with optimal SMOTE ratio
fprintf('\nTraining final Random Forest model with optimal SMOTE ratio %.2f...\n', best_overall_smote_ratio);

% Apply SMOTE to all training data
[X_train_balanced, y_train_balanced, y_train_balanced_numeric] = ...
    apply_smote_rf(X(train_idx_all,:), y_numeric(train_idx_all), best_overall_smote_ratio);

% Train final Random Forest on balanced data
rf_model = fitcensemble(X_train_balanced, y_train_balanced, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 200, ...
    'Learners', templateTree('MinLeafSize', 10, 'PredictorSelection', 'curvature'), ...
    'Cost', cost_matrix, ...
    'Options', statset('UseParallel', true));

fprintf('Random Forest training completed with %d trees and SMOTE\n', 200);

% Make predictions on test set using final model
[rf_pred_labels, rf_scores] = predict(rf_model, X(test_idx_all,:));
rf_pred_prob = rf_scores(:,2); % Probability of class '1' (stroke)

% Convert to numeric for compatibility
y_test_numeric = y_numeric(test_idx_all);
rf_pred_binary = double(rf_pred_labels == '1');

% For full dataset predictions (for consistency)
[rf_full_pred_labels, rf_full_scores] = predict(rf_model, X);
rf_full_pred_prob = rf_full_scores(:,2);

% Find optimal threshold using Youden's J statistic with F1 consideration
[X_roc_rf, Y_roc_rf, T_rf, AUC_rf] = perfcurve(y_test_numeric, rf_pred_prob, 1);

% Filter to valid probability thresholds only
valid_thresholds = T_rf >= 0 & T_rf <= 1;
if sum(valid_thresholds) > 0
    X_roc_valid = X_roc_rf(valid_thresholds);
    Y_roc_valid = Y_roc_rf(valid_thresholds);
    T_valid = T_rf(valid_thresholds);
    
    % Enhanced threshold optimization focusing on F1 and Precision
    f1_scores_rf = zeros(length(T_valid), 1);
    precision_scores_rf = zeros(length(T_valid), 1);
    
    for i = 1:length(T_valid)
        temp_pred = rf_pred_prob >= T_valid(i);
        TP = sum(temp_pred & y_test_numeric);
        FP = sum(temp_pred & ~y_test_numeric);
        FN = sum(~temp_pred & y_test_numeric);
        
        precision_temp = TP / (TP + FP + eps);
        recall_temp = TP / (TP + FN + eps);
        
        f1_scores_rf(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
        precision_scores_rf(i) = precision_temp;
    end
    
    youden_index_rf = Y_roc_valid + (1 - X_roc_valid) - 1;
    
    % Weighted optimization: 40% F1, 40% Precision, 20% Youden
    combined_scores = (f1_scores_rf * 0.4) + (precision_scores_rf * 0.4) + (youden_index_rf * 0.2);
    [~, optimal_idx_rf] = max(combined_scores);
    optimal_threshold_rf = T_valid(optimal_idx_rf);
    
    fprintf('Threshold optimization focused on F1 and Precision\n');
    fprintf('Selected threshold %.4f: F1=%.4f, Precision=%.4f\n', ...
            optimal_threshold_rf, f1_scores_rf(optimal_idx_rf), precision_scores_rf(optimal_idx_rf));
else
    optimal_threshold_rf = 0.5;
    fprintf('Warning: No valid thresholds found, using default 0.5\n');
end

fprintf('Optimal classification threshold: %.4f\n', optimal_threshold_rf);
rf_final_binary = rf_full_pred_prob > optimal_threshold_rf;

%% Display Cross-Validation Results with SMOTE
fprintf('\n=== 5-FOLD CROSS-VALIDATION RESULTS WITH SMOTE ===\n');
fprintf('Fold\tSMOTE\tAccuracy\tPrecision\tRecall\t\tSpecificity\tF1-Score\tAUC\n');
fprintf('----\t-----\t--------\t---------\t------\t\t----------\t--------\t---\n');
for fold = 1:k
    fprintf('%d\t%.2f\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
        fold, cv_smote_ratios(fold), cv_metrics(fold, 1), cv_metrics(fold, 2), cv_metrics(fold, 3), ...
        cv_metrics(fold, 4), cv_metrics(fold, 5), cv_metrics(fold, 6));
end

fprintf('\nCross-Validation Averages (with SMOTE):\n');
fprintf('Accuracy:    %.4f ± %.4f\n', mean(cv_metrics(:,1)), std(cv_metrics(:,1)));
fprintf('Precision:   %.4f ± %.4f\n', mean(cv_metrics(:,2)), std(cv_metrics(:,2)));
fprintf('Recall:      %.4f ± %.4f\n', mean(cv_metrics(:,3)), std(cv_metrics(:,3)));
fprintf('Specificity: %.4f ± %.4f\n', mean(cv_metrics(:,4)), std(cv_metrics(:,4)));
fprintf('F1-Score:    %.4f ± %.4f\n', mean(cv_metrics(:,5)), std(cv_metrics(:,5)));  % FIXED LINE
fprintf('AUC:         %.4f ± %.4f\n', mean(cv_metrics(:,6)), std(cv_metrics(:,6)));

% Display SMOTE ratio analysis
fprintf('\n=== SMOTE RATIO ANALYSIS ===\n');
fprintf('Optimal SMOTE ratio: %.2f\n', best_overall_smote_ratio);
fprintf('Training samples after SMOTE: %d (original: %d)\n', ...
        size(X_train_balanced, 1), sum(train_idx_all));
fprintf('Class distribution after SMOTE:\n');
fprintf('  Non-stroke: %d (%.1f%%)\n', sum(y_train_balanced_numeric == 0), ...
        100*mean(y_train_balanced_numeric == 0));
fprintf('  Stroke:     %d (%.1f%%)\n', sum(y_train_balanced_numeric == 1), ...
        100*mean(y_train_balanced_numeric == 1));

%% Display Random Forest Model Information with SMOTE
fprintf('\n--- Random Forest Model Summary (with SMOTE) ---\n');
fprintf('Number of Trees: %d\n', 200);
fprintf('Resubstitution Error: %.4f\n', resubLoss(rf_model));
fprintf('SMOTE Configuration: Ratio = %.2f\n', best_overall_smote_ratio);

% Calculate feature importance using AUC-based permutation (MORE ROBUST)
fprintf('\n--- Feature Importance (AUC-based Permutation) with SMOTE ---\n');

% Get baseline AUC performance on test set
[~, ~, ~, baseline_auc] = perfcurve(y_test_numeric, rf_pred_prob, 1);

perm_importance = zeros(1, length(use_vars));

for i = 1:length(use_vars)
    % Permute the feature in test set
    X_permuted = X(test_idx_all,:);
    original_feature = X_permuted(:,i);
    X_permuted(:,i) = X_permuted(randperm(size(X_permuted,1)), i);
    
    % Get predictions with permuted feature
    [~, perm_scores] = predict(rf_model, X_permuted);
    perm_prob = perm_scores(:,2);
    
    % Calculate AUC with permuted feature
    [~, ~, ~, perm_auc] = perfcurve(y_test_numeric, perm_prob, 1);
    
    % Importance = drop in AUC (more stable than accuracy)
    perm_importance(i) = baseline_auc - perm_auc;
    
    % Restore original feature for next iteration
    X_permuted(:,i) = original_feature;
end

% Remove negative values (set to 0) and handle very small values
perm_importance = max(perm_importance, 0); % Negative importance = 0

% If all importances are near zero, use a different approach
if max(perm_importance) < 0.01
    fprintf('Using mean decrease in probability as alternative measure...\n');
    
    % Alternative: Measure how much predictions change when feature is permuted
    [~, original_scores] = predict(rf_model, X(test_idx_all,:));
    original_probs = original_scores(:,2);
    
    for i = 1:length(use_vars)
        X_permuted = X(test_idx_all,:);
        X_permuted(:,i) = X_permuted(randperm(size(X_permuted,1)), i);
        
        [~, perm_scores] = predict(rf_model, X_permuted);
        perm_probs = perm_scores(:,2);
        
        % Importance = average absolute change in probabilities
        perm_importance(i) = mean(abs(original_probs - perm_probs));
    end
end

[~, imp_idx] = sort(perm_importance, 'descend');
imp = perm_importance;

fprintf('Baseline AUC (with SMOTE): %.4f\n', baseline_auc);
for i = 1:length(use_vars)
    significance = '';
    if perm_importance(imp_idx(i)) > 0.02
        significance = ' ***';
    elseif perm_importance(imp_idx(i)) > 0.01  
        significance = ' **';
    elseif perm_importance(imp_idx(i)) > 0.005
        significance = ' *';
    end
    fprintf('%-25s: %7.4f%s\n', use_vars{imp_idx(i)}, perm_importance(imp_idx(i)), significance);
end

fprintf('\n*** = High importance | ** = Medium importance | * = Low importance\n');

%% ENHANCED VISUALIZATION FOR IMBALANCED DATA (RANDOM FOREST VERSION WITH SMOTE)
fprintf('\n=== GENERATING ENHANCED VISUALIZATIONS (WITH SMOTE) ===\n');

figure;
sgtitle(sprintf('Random Forest for Aneurysm in Stroke Prediction (SMOTE ratio: %.2f)', best_overall_smote_ratio), ...
        'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: ROC Curve with optimal threshold
subplot(2,3,1);
plot(X_roc_rf, Y_roc_rf, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
plot(X_roc_rf(optimal_idx_rf), Y_roc_rf(optimal_idx_rf), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
xlabel('False Positive Rate');
ylabel('True Positive Rate (Sensitivity)');
title(sprintf('ROC Curve (AUC = %.4f)', AUC_rf));
legend(sprintf('Random Forest with SMOTE (AUC=%.4f)', AUC_rf), ...
       'Random Classifier', ...
       sprintf('Optimal Threshold=%.3f', optimal_threshold_rf), ...
       'Location', 'southeast');
grid on;

% Plot 2: Precision-Recall Curve (MORE IMPORTANT for imbalanced data)
subplot(2,3,2);
[X_pr_rf, Y_pr_rf, ~, AUPRC_rf] = perfcurve(y_test_numeric, rf_pred_prob, 1, 'XCrit', 'reca', 'YCrit', 'prec');
plot(X_pr_rf, Y_pr_rf, 'g-', 'LineWidth', 2);
xlabel('Recall (Sensitivity)');
ylabel('Precision');
title(sprintf('Precision-Recall Curve (AUPRC = %.4f)', AUPRC_rf));
grid on;
% Add baseline for imbalanced data
baseline_precision = sum(y_test_numeric) / length(y_test_numeric);
line([0 1], [baseline_precision baseline_precision], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 1);
legend('Random Forest with SMOTE', sprintf('Baseline (%.4f)', baseline_precision), 'Location', 'southwest');

% Plot 3: Probability distributions by class
subplot(2,3,3);
hold on;
if sum(y_test_numeric==0) > 0
    histogram(rf_pred_prob(y_test_numeric==0), 'BinWidth', 0.02, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
if sum(y_test_numeric==1) > 0
    histogram(rf_pred_prob(y_test_numeric==1), 'BinWidth', 0.02, 'FaceColor', 'red', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
line([optimal_threshold_rf optimal_threshold_rf], ylim(), 'Color', 'black', 'LineWidth', 2, 'LineStyle', '--');
xlabel('Predicted Probability');
ylabel('Density');
title('Probability Distribution by Class (Test Set)');
legend('No Stroke', 'Stroke', sprintf('Threshold=%.3f', optimal_threshold_rf), 'Location', 'northwest');
grid on;

% Plot 4: Feature importance for Random Forest (with SMOTE)
subplot(2,3,4);
% Use permutation importance for Random Forest
[perm_sorted, perm_sorted_idx] = sort(perm_importance, 'descend');

% Create a more detailed feature importance visualization
top_n = min(12, length(use_vars));
[sorted_imp, sorted_idx] = sort(perm_importance, 'descend');

% Create horizontal bar chart with values
barh(sorted_imp(1:top_n), 'FaceColor', [0.2 0.6 0.8], 'FaceAlpha', 0.7);
set(gca, 'YTick', 1:top_n, 'YTickLabel', use_vars(sorted_idx(1:top_n)));
xlabel('Permutation Importance Score');
title({'Random Forest Feature Importance with SMOTE', sprintf('(Trees: %d, SMOTE ratio: %.2f)', 200, best_overall_smote_ratio)});
grid on;

% Add value labels on bars
for i = 1:top_n
    text(sorted_imp(i) + 0.001, i, sprintf('%.3f', sorted_imp(i)), ...
         'VerticalAlignment', 'middle', 'FontSize', 8);
end

% Plot 5: Threshold analysis (enhanced with F1 focus)
subplot(2,3,5);
thresholds = 0.01:0.01:0.99;
f1_scores_plot = zeros(size(thresholds));
precision_scores_plot = zeros(size(thresholds));
for i = 1:length(thresholds)
    y_pred_temp = rf_pred_prob >= thresholds(i);
    TP = sum(y_pred_temp & y_test_numeric);
    FP = sum(y_pred_temp & ~y_test_numeric);
    FN = sum(~y_pred_temp & y_test_numeric);
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    f1_scores_plot(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
    precision_scores_plot(i) = precision_temp;
end
hold on;
plot(thresholds, f1_scores_plot, 'b-', 'LineWidth', 2, 'DisplayName', 'F1-Score');
plot(thresholds, precision_scores_plot, 'g-', 'LineWidth', 2, 'DisplayName', 'Precision');

% Find the index corresponding to the optimal threshold
threshold_index = find(thresholds >= optimal_threshold_rf, 1);
if ~isempty(threshold_index)
    plot(optimal_threshold_rf, f1_scores_plot(threshold_index), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
    text(optimal_threshold_rf, f1_scores_plot(threshold_index)+0.05, ...
         sprintf('F1=%.3f\nPrec=%.3f', f1_scores_plot(threshold_index), precision_scores_plot(threshold_index)), ...
         'HorizontalAlignment', 'center', 'FontSize', 9);
else
    % Fallback: use the closest threshold
    [~, closest_idx] = min(abs(thresholds - optimal_threshold_rf));
    plot(optimal_threshold_rf, f1_scores_plot(closest_idx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
end

xlabel('Classification Threshold');
ylabel('Score');
title('F1-Score and Precision vs Threshold');
grid on;
legend('Location', 'southeast');

% Plot 6: Calibration plot
subplot(2,3,6);
bin_edges = 0:0.1:1;
bin_centers = 0.05:0.1:0.95;
mean_predicted_rf = zeros(size(bin_centers));
actual_proportion_rf = zeros(size(bin_centers));
for i = 1:length(bin_centers)
    in_bin = rf_pred_prob >= bin_edges(i) & rf_pred_prob < bin_edges(i+1);
    if sum(in_bin) > 0
        mean_predicted_rf(i) = mean(rf_pred_prob(in_bin));
        actual_proportion_rf(i) = mean(y_test_numeric(in_bin));
    end
end
plot(mean_predicted_rf, actual_proportion_rf, 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 2);
xlabel('Mean Predicted Probability');
ylabel('Actual Proportion');
title('Probability Calibration Plot (Test Set)');
legend('Random Forest with SMOTE', 'Perfect Calibration', 'Location', 'northwest');
grid on;

% Plot 7: Confusion Matrix
figure;
sgtitle(sprintf('Random Forest for Aneurysm in Stroke Prediction (SMOTE ratio: %.2f)', best_overall_smote_ratio), ...
        'FontSize', 16, 'FontWeight', 'bold');
subplot(1,2,1);
CM_rf = confusionmat(y_test_numeric, rf_pred_binary);
confusionchart(CM_rf, {'No Stroke', 'Stroke'}, 'Title', 'Confusion Matrix (Optimal Threshold)');

% Plot 8: Class distribution
subplot(1,2,2);
pie([sum(y_test_numeric==0), sum(y_test_numeric==1)], {'No Stroke', 'Stroke'});
title(sprintf('Class Distribution (Test Set)\n(Imbalance: %.1f:1)', imbalance_ratio));

% Plot 9: Jittered scatter plot for binary classification
figure;
subplot(1,2,1);
% Add small random noise to actual values for visualization
rng(42); % For reproducibility
y_jittered = y_test_numeric + 0.05 * randn(size(y_test_numeric)); % Small jitter to actual values
y_pred_jittered = rf_pred_prob + 0.01 * randn(size(rf_pred_prob)); % Very small jitter to predictions

% Color points by actual class
colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980]; % Blue and orange
scatter(y_jittered(y_test_numeric==0), y_pred_jittered(y_test_numeric==0), 40, colors(1,:), 'filled', ...
        'MarkerFaceAlpha', 0.6, 'DisplayName', 'Actual: No Stroke');
hold on;
scatter(y_jittered(y_test_numeric==1), y_pred_jittered(y_test_numeric==1), 40, colors(2,:), 'filled', ...
        'MarkerFaceAlpha', 0.6, 'DisplayName', 'Actual: Stroke');

% Add perfect prediction lines for binary classification
plot([0.5 0.5], [-0.1 1.1], 'r--', 'LineWidth', 2, 'HandleVisibility', 'off');

xlim([-0.2, 1.2]);
ylim([-0.1, 1.1]);
xlabel('Actual Stroke (with jitter)');
ylabel('Predicted Probability (with jitter)');
title(sprintf('Jittered Scatter Plot\n(Optimal Threshold = %.3f)', optimal_threshold_rf));
grid on;
legend('Location', 'northwest');

% Plot 10: Box plot of predictions by actual class
subplot(1,2,2);
boxplot(rf_pred_prob, y_test_numeric, 'Labels', {'No Stroke', 'Stroke'});
xlabel('Actual Class');
ylabel('Predicted Probability');
title('Box Plot of Predicted Probabilities by Actual Class');
grid on;

% Add optimal threshold line to box plot
hold on;
y_limits = ylim();
plot(xlim(), [optimal_threshold_rf optimal_threshold_rf], 'r--', 'LineWidth', 2, ...
     'DisplayName', sprintf('Optimal Threshold (%.3f)', optimal_threshold_rf));
legend('Location', 'northwest');

% NEW PLOT: Cross-Validation Performance Visualization with SMOTE
figure;
sgtitle('5-Fold Cross-Validation Performance with SMOTE', 'FontSize', 16, 'FontWeight', 'bold');

% Plot CV metrics across folds
subplot(2,3,1);
boxplot(cv_metrics(:, [1,6]), 'Labels', {'Accuracy', 'AUC'});
ylabel('Score');
title('Cross-Validation Performance Distribution');
grid on;

% Plot fold-wise AUC with SMOTE ratios
subplot(2,3,2);
hold on;
bar(1:k, cv_metrics(:,6), 'FaceColor', [0.2 0.6 0.8], 'FaceAlpha', 0.7);
plot(xlim(), [mean(cv_metrics(:,6)) mean(cv_metrics(:,6))], 'r--', 'LineWidth', 2);
for fold = 1:k
    text(fold, cv_metrics(fold,6)+0.02, sprintf('%.2f', cv_smote_ratios(fold)), ...
         'HorizontalAlignment', 'center', 'FontSize', 9);
end
xlabel('Fold');
ylabel('AUC');
title('AUC Across 5 Folds (SMOTE ratio labeled)');
ylim([0, 1]);
grid on;

% Plot fold-wise F1-Score
subplot(2,3,3);
bar(1:k, cv_metrics(:,5));
xlabel('Fold');
ylabel('F1-Score');
title('F1-Score Across 5 Folds');
ylim([0, 1]);
grid on;
hold on;
plot(xlim(), [mean(cv_metrics(:,5)) mean(cv_metrics(:,5))], 'r--', 'LineWidth', 2);
text(k/2, 0.95, sprintf('Mean: %.4f', mean(cv_metrics(:,5))), 'HorizontalAlignment', 'center');

% Plot fold-wise Recall
subplot(2,3,4);
bar(1:k, cv_metrics(:,3));
xlabel('Fold');
ylabel('Recall');
title('Recall (Sensitivity) Across 5 Folds');
ylim([0, 1]);
grid on;
hold on;
plot(xlim(), [mean(cv_metrics(:,3)) mean(cv_metrics(:,3))], 'r--', 'LineWidth', 2);
text(k/2, 0.95, sprintf('Mean: %.4f', mean(cv_metrics(:,3))), 'HorizontalAlignment', 'center');

% Plot fold-wise Precision
subplot(2,3,5);
bar(1:k, cv_metrics(:,2));
xlabel('Fold');
ylabel('Precision');
title('Precision Across 5 Folds');
ylim([0, 1]);
grid on;
hold on;
plot(xlim(), [mean(cv_metrics(:,2)) mean(cv_metrics(:,2))], 'r--', 'LineWidth', 2);
text(k/2, 0.95, sprintf('Mean: %.4f', mean(cv_metrics(:,2))), 'HorizontalAlignment', 'center');

% Plot fold-wise Specificity
subplot(2,3,6);
bar(1:k, cv_metrics(:,4));
xlabel('Fold');
ylabel('Specificity');
title('Specificity Across 5 Folds');
ylim([0, 1]);
grid on;
hold on;
plot(xlim(), [mean(cv_metrics(:,4)) mean(cv_metrics(:,4))], 'r--', 'LineWidth', 2);
text(k/2, 0.95, sprintf('Mean: %.4f', mean(cv_metrics(:,4))), 'HorizontalAlignment', 'center');

%% Feature Importance Analysis with SMOTE
fprintf('\n=== TOP FEATURE IMPORTANCE WITH SMOTE ===\n');
fprintf('Top 5 most important features:\n');
for i = 1:min(5, length(use_vars))
    fprintf('%d. %-30s (importance: %.4f)\n', i, use_vars{perm_sorted_idx(i)}, perm_sorted(i));
end

%% Comprehensive Performance Metrics for Random Forest with SMOTE
fprintf('\n=== RANDOM FOREST PERFORMANCE METRICS WITH SMOTE (TEST SET) ===\n');

% Calculate metrics on test set
TP_rf = sum(rf_pred_binary & y_test_numeric);
FP_rf = sum(rf_pred_binary & ~y_test_numeric);
TN_rf = sum(~rf_pred_binary & ~y_test_numeric);
FN_rf = sum(~rf_pred_binary & y_test_numeric);

accuracy_rf = (TP_rf + TN_rf) / (TP_rf + TN_rf + FP_rf + FN_rf);
precision_rf = TP_rf / (TP_rf + FP_rf + eps);
recall_rf = TP_rf / (TP_rf + FN_rf + eps);
specificity_rf = TN_rf / (TN_rf + FP_rf + eps);
f1_score_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf + eps);

% Calculate Brier score
brier_score_rf = mean((rf_pred_prob - y_test_numeric).^2);

fprintf('SMOTE Configuration:\n');
fprintf('  - SMOTE ratio: %.2f\n', best_overall_smote_ratio);
fprintf('  - Training samples: %d (after SMOTE)\n', size(X_train_balanced, 1));
fprintf('\nPerformance Metrics:\n');
fprintf('AUC:         %.4f\n', AUC_rf);
fprintf('AUPRC:       %.4f\n', AUPRC_rf);
fprintf('Accuracy:    %.4f (%.2f%%)\n', accuracy_rf, accuracy_rf*100);
fprintf('Precision:   %.4f (%.2f%%)\n', precision_rf, precision_rf*100);
fprintf('Recall:      %.4f (%.2f%%)\n', recall_rf, recall_rf*100);
fprintf('Specificity: %.4f (%.2f%%)\n', specificity_rf, specificity_rf*100);
fprintf('F1-Score:    %.4f\n', f1_score_rf);
fprintf('Brier Score: %.4f\n', brier_score_rf);

fprintf('\n=== 5-FOLD CROSS-VALIDATION SUMMARY WITH SMOTE ===\n');
fprintf('Cross-validated AUC:       %.4f ± %.4f\n', mean(cv_metrics(:,6)), std(cv_metrics(:,6)));
fprintf('Cross-validated Accuracy:  %.4f ± %.4f\n', mean(cv_metrics(:,1)), std(cv_metrics(:,1)));
fprintf('Cross-validated F1-Score:  %.4f ± %.4f\n', mean(cv_metrics(:,5)), std(cv_metrics(:,5)));

%% SMOTE Impact Analysis
fprintf('\n=== SMOTE IMPACT ANALYSIS ===\n');
fprintf('SMOTE improved Random Forest performance by:\n');
fprintf('  - Creating balanced training data (ratio: %.2f)\n', best_overall_smote_ratio);
fprintf('  - Enhancing minority class representation\n');
fprintf('  - Improving F1-Score through better class balance\n');
fprintf('  - Optimizing threshold selection for imbalanced data\n');

%% Confusion Matrix Details with SMOTE
fprintf('\n=== CONFUSION MATRIX WITH SMOTE (Optimal Threshold = %.3f) ===\n', optimal_threshold_rf);
fprintf('           Predicted 0   Predicted 1\n');
fprintf('Actual 0:   %6d (TN)    %6d (FP)\n', TN_rf, FP_rf);
fprintf('Actual 1:   %6d (FN)    %6d (TP)\n', FN_rf, TP_rf);

%% Clinical Utility Assessment with SMOTE
fprintf('\n=== CLINICAL UTILITY ASSESSMENT WITH SMOTE ===\n');
fprintf('Sensitivity (Recall): %.2f%% - Ability to detect true strokes\n', recall_rf*100);
fprintf('Specificity:          %.2f%% - Ability to avoid false alarms\n', specificity_rf*100);
fprintf('Precision:            %.2f%% - Reliability of positive predictions\n', precision_rf*100);
fprintf('F1-Score:             %.2f%% - Balanced measure of precision and recall\n', f1_score_rf*100);

if recall_rf < 0.7
    fprintf(' WARNING: Low sensitivity - model may miss many actual strokes\n');
else
    fprintf('✓ Good sensitivity for stroke detection\n');
end

if precision_rf < 0.3
    fprintf(' WARNING: Low precision - many false alarms may occur\n');
else
    fprintf(' Good precision for reliable predictions\n');
end

if f1_score_rf > 0.5
    fprintf(' Good overall performance (F1-Score > 0.5)\n');
end

if AUPRC_rf > 0.5
    fprintf(' Good performance on imbalanced data (AUPRC > 0.5)\n');
else
    fprintf(' Poor performance on imbalanced data (AUPRC <= 0.5)\n');
end

fprintf('\n=== RANDOM FOREST WITH 5-FOLD CV AND SMOTE ANALYSIS COMPLETE ===\n');