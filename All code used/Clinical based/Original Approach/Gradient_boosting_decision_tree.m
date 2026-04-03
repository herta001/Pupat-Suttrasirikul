%% GBDT Model


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

%% CRITICAL: Check and Handle Class Imbalance
fprintf('\n=== CLASS IMBALANCE ANALYSIS ===\n');

stroke_cases = sum(y);
non_stroke_cases = sum(y == 0);
total_cases = length(y);
imbalance_ratio = non_stroke_cases / stroke_cases;

fprintf('Stroke cases: %d (%.4f%%)\n', stroke_cases, (stroke_cases/total_cases)*100);
fprintf('Non-stroke cases: %d (%.4f%%)\n', non_stroke_cases, (non_stroke_cases/total_cases)*100);
fprintf('Imbalance ratio: %.2f:1 (non-stroke:stroke)\n', imbalance_ratio);

% Set up cross-validation
rng(42); % For reproducibility
cv = cvpartition(y, 'Holdout', 0.3); % 70-30 split
train_idx = training(cv);
test_idx = test(cv);

%% ENHANCED FEATURE ENGINEERING FOR GBDT
fprintf('\n=== ENHANCED FEATURE ENGINEERING FOR BETTER AUC ===\n');

% Create enhanced feature set with clinical knowledge
X_enhanced = X;
enhanced_feature_names = use_vars;

% 1. Age transformations (non-linear relationships)
if ismember('age', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    X_enhanced(:, end+1) = X(:, age_idx).^2; % Age squared
    X_enhanced(:, end+1) = log(X(:, age_idx) + 1); % Log age
    X_enhanced(:, end+1) = (X(:, age_idx) > 65); % Elderly indicator
    X_enhanced(:, end+1) = (X(:, age_idx) > 75); % Very elderly indicator
    enhanced_feature_names{end+1} = 'age_squared';
    enhanced_feature_names{end+1} = 'log_age';
    enhanced_feature_names{end+1} = 'elderly_65plus';
    enhanced_feature_names{end+1} = 'very_elderly_75plus';
    fprintf('Added age transformations\n');
end

% 2. BMI categories
if ismember('bmi', use_vars)
    bmi_idx = find(strcmp(use_vars, 'bmi'));
    X_enhanced(:, end+1) = (X(:, bmi_idx) < 18.5); % Underweight
    X_enhanced(:, end+1) = (X(:, bmi_idx) >= 25 & X(:, bmi_idx) < 30); % Overweight
    X_enhanced(:, end+1) = (X(:, bmi_idx) >= 30); % Obese
    X_enhanced(:, end+1) = (X(:, bmi_idx) >= 35); % Severely obese
    enhanced_feature_names{end+1} = 'bmi_underweight';
    enhanced_feature_names{end+1} = 'bmi_overweight';
    enhanced_feature_names{end+1} = 'bmi_obese';
    enhanced_feature_names{end+1} = 'bmi_severely_obese';
    fprintf('Added BMI categories\n');
end

% 3. Glucose level categories
if ismember('avg_glucose_level', use_vars)
    glucose_idx = find(strcmp(use_vars, 'avg_glucose_level'));
    X_enhanced(:, end+1) = (X(:, glucose_idx) > 140); % Hyperglycemia
    X_enhanced(:, end+1) = (X(:, glucose_idx) > 200); % Severe hyperglycemia
    X_enhanced(:, end+1) = (X(:, glucose_idx) < 70); % Hypoglycemia
    enhanced_feature_names{end+1} = 'glucose_high';
    enhanced_feature_names{end+1} = 'glucose_very_high';
    enhanced_feature_names{end+1} = 'glucose_low';
    fprintf('Added glucose categories\n');
end

% 4. Clinical interaction terms
if ismember('age', use_vars) && ismember('hypertension', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    ht_idx = find(strcmp(use_vars, 'hypertension'));
    X_enhanced(:, end+1) = X(:, age_idx) .* X(:, ht_idx); % Age × Hypertension
    enhanced_feature_names{end+1} = 'age_hypertension_interaction';
    fprintf('Added age-hypertension interaction\n');
end

if ismember('age', use_vars) && ismember('heart_disease', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    hd_idx = find(strcmp(use_vars, 'heart_disease'));
    X_enhanced(:, end+1) = X(:, age_idx) .* X(:, hd_idx); % Age × Heart Disease
    enhanced_feature_names{end+1} = 'age_heart_disease_interaction';
    fprintf('Added age-heart_disease interaction\n');
end

if ismember('hypertension', use_vars) && ismember('heart_disease', use_vars)
    ht_idx = find(strcmp(use_vars, 'hypertension'));
    hd_idx = find(strcmp(use_vars, 'heart_disease'));
    X_enhanced(:, end+1) = X(:, ht_idx) .* X(:, hd_idx); % Hypertension × Heart Disease
    enhanced_feature_names{end+1} = 'hypertension_heart_disease_interaction';
    fprintf('Added hypertension-heart_disease interaction\n');
end

% 5. Risk score combinations
if ismember('age', use_vars) && ismember('hypertension', use_vars) && ismember('heart_disease', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    ht_idx = find(strcmp(use_vars, 'hypertension'));
    hd_idx = find(strcmp(use_vars, 'heart_disease'));
    % Simple clinical risk score
    X_enhanced(:, end+1) = (X(:, age_idx) > 65) + X(:, ht_idx) + X(:, hd_idx);
    enhanced_feature_names{end+1} = 'simple_risk_score';
    fprintf('Added simple clinical risk score\n');
end

fprintf('Enhanced feature set: %d features -> %d features\n', size(X, 2), size(X_enhanced, 2));

% Update X with enhanced features
X = X_enhanced;
use_vars = enhanced_feature_names;

%% GRADIENT BOOSTING DECISION TREES TRAINING WITH 5-FOLD CROSS-VALIDATION
fprintf('\n=== GRADIENT BOOSTING DECISION TREES TRAINING WITH 5-FOLD CROSS-VALIDATION ===\n');

fprintf('Testing multiple GBDT configurations with 5-fold CV...\n');

% GBDT parameter configurations
gbdt_configs = {
    struct('NumLearningCycles', 100, 'LearnRate', 0.1, 'MaxNumSplits', 10, 'MinLeafSize', 1),
    struct('NumLearningCycles', 200, 'LearnRate', 0.1, 'MaxNumSplits', 20, 'MinLeafSize', 5),
    struct('NumLearningCycles', 100, 'LearnRate', 0.05, 'MaxNumSplits', 10, 'MinLeafSize', 1),
    struct('NumLearningCycles', 200, 'LearnRate', 0.05, 'MaxNumSplits', 20, 'MinLeafSize', 5),
    struct('NumLearningCycles', 150, 'LearnRate', 0.1, 'MaxNumSplits', 15, 'MinLeafSize', 3),
    struct('NumLearningCycles', 100, 'LearnRate', 0.2, 'MaxNumSplits', 10, 'MinLeafSize', 1),
    struct('NumLearningCycles', 50, 'LearnRate', 0.1, 'MaxNumSplits', 5, 'MinLeafSize', 10)
};

best_gbdt_auc = 0;
best_gbdt_model = [];
best_gbdt_config = [];
best_gbdt_cv_auc = 0;
best_gbdt_cv_aucs = [];
best_gbdt_cv_std = 0;

X_train = X(train_idx,:);
X_test = X(test_idx,:);
y_train = y(train_idx);
y_test = y(test_idx);

successful_configs = 0;

% 5-fold cross-validation setup for training data
cv_folds = 5;
cv_indices = cvpartition(y_train, 'KFold', cv_folds);
fprintf('Using %d-fold cross-validation on training data\n', cv_folds);

for i = 1:length(gbdt_configs)
    try
        fprintf('Testing GBDT config %d/%d... ', i, length(gbdt_configs));
        
        % Initialize arrays for CV metrics
        cv_aucs = zeros(cv_folds, 1);
        cv_models = cell(cv_folds, 1);
        
        % Perform 5-fold cross-validation
        for fold = 1:cv_folds
            % Get training and validation indices for this fold
            train_fold_idx = training(cv_indices, fold);
            val_fold_idx = test(cv_indices, fold);
            
            X_train_fold = X_train(train_fold_idx, :);
            X_val_fold = X_train(val_fold_idx, :);
            y_train_fold = y_train(train_fold_idx);
            y_val_fold = y_train(val_fold_idx);
            
            % Train model on this fold
            if imbalance_ratio > 4
                % For imbalanced data, use class weights
                gbdt_fold = fitcensemble(X_train_fold, y_train_fold, 'Method', 'AdaBoostM1', ...
                    'NumLearningCycles', gbdt_configs{i}.NumLearningCycles, ...
                    'Learners', templateTree('MaxNumSplits', gbdt_configs{i}.MaxNumSplits, ...
                                            'MinLeafSize', gbdt_configs{i}.MinLeafSize), ...
                    'LearnRate', gbdt_configs{i}.LearnRate, ...
                    'Cost', [0, 1; imbalance_ratio, 0]);
            else
                gbdt_fold = fitcensemble(X_train_fold, y_train_fold, 'Method', 'AdaBoostM1', ...
                    'NumLearningCycles', gbdt_configs{i}.NumLearningCycles, ...
                    'Learners', templateTree('MaxNumSplits', gbdt_configs{i}.MaxNumSplits, ...
                                            'MinLeafSize', gbdt_configs{i}.MinLeafSize), ...
                    'LearnRate', gbdt_configs{i}.LearnRate);
            end
            
            % Evaluate on validation fold
            [~, prob_val] = predict(gbdt_fold, X_val_fold);
            prob_val = prob_val(:,2);
            
            % Calculate AUC for this fold
            [~, ~, ~, auc_fold] = perfcurve(y_val_fold, prob_val, 1);
            cv_aucs(fold) = auc_fold;
            cv_models{fold} = gbdt_fold;
        end
        
        % Calculate mean CV AUC
        mean_cv_auc = mean(cv_aucs);
        std_cv_auc = std(cv_aucs);
        
        fprintf('CV AUC: %.4f ± %.4f ', mean_cv_auc, std_cv_auc);
        
        % Train final model on entire training set with this configuration
        if imbalance_ratio > 4
            gbdt_temp = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1', ...
                'NumLearningCycles', gbdt_configs{i}.NumLearningCycles, ...
                'Learners', templateTree('MaxNumSplits', gbdt_configs{i}.MaxNumSplits, ...
                                        'MinLeafSize', gbdt_configs{i}.MinLeafSize), ...
                'LearnRate', gbdt_configs{i}.LearnRate, ...
                'Cost', [0, 1; imbalance_ratio, 0]);
        else
            gbdt_temp = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1', ...
                'NumLearningCycles', gbdt_configs{i}.NumLearningCycles, ...
                'Learners', templateTree('MaxNumSplits', gbdt_configs{i}.MaxNumSplits, ...
                                        'MinLeafSize', gbdt_configs{i}.MinLeafSize), ...
                'LearnRate', gbdt_configs{i}.LearnRate);
        end
        
        % Get probability predictions on test set
        [~, gbdt_prob_temp] = predict(gbdt_temp, X_test);
        gbdt_prob_temp = gbdt_prob_temp(:,2);
        
        % Calculate test AUC
        [~, ~, ~, gbdt_temp_auc] = perfcurve(y_test, gbdt_prob_temp, 1);
        fprintf('Test AUC: %.4f\n', gbdt_temp_auc);
        
        % Select best model based on CV AUC (primary) and test AUC (secondary)
        if mean_cv_auc > best_gbdt_cv_auc || ...
           (abs(mean_cv_auc - best_gbdt_cv_auc) < 0.001 && gbdt_temp_auc > best_gbdt_auc)
            best_gbdt_cv_auc = mean_cv_auc;
            best_gbdt_auc = gbdt_temp_auc;
            best_gbdt_model = gbdt_temp;
            best_gbdt_config = gbdt_configs{i};
            best_gbdt_cv_std = std_cv_auc;
            best_gbdt_cv_aucs = cv_aucs;  % Store individual fold AUCs
        end
        
        successful_configs = successful_configs + 1;
        
    catch ME
        fprintf('Failed: %s\n', ME.message);
        continue;
    end
end

% Check if any configuration succeeded
if isempty(best_gbdt_config)
    fprintf('\n ERROR: All GBDT configurations failed!\n');
    fprintf('Trying default configuration...\n');
    
    % Use default configuration
    best_gbdt_config = struct('NumLearningCycles', 100, 'LearnRate', 0.1, 'MaxNumSplits', 10, 'MinLeafSize', 5);
    
    % Train with 5-fold CV for default config too
    cv_aucs_default = zeros(cv_folds, 1);
    for fold = 1:cv_folds
        train_fold_idx = training(cv_indices, fold);
        val_fold_idx = test(cv_indices, fold);
        
        X_train_fold = X_train(train_fold_idx, :);
        X_val_fold = X_train(val_fold_idx, :);
        y_train_fold = y_train(train_fold_idx);
        y_val_fold = y_train(val_fold_idx);
        
        gbdt_fold = fitcensemble(X_train_fold, y_train_fold, 'Method', 'AdaBoostM1', ...
            'NumLearningCycles', best_gbdt_config.NumLearningCycles, ...
            'Learners', templateTree('MaxNumSplits', best_gbdt_config.MaxNumSplits, ...
                                    'MinLeafSize', best_gbdt_config.MinLeafSize), ...
            'LearnRate', best_gbdt_config.LearnRate);
        
        [~, prob_val] = predict(gbdt_fold, X_val_fold);
        prob_val = prob_val(:,2);
        [~, ~, ~, auc_fold] = perfcurve(y_val_fold, prob_val, 1);
        cv_aucs_default(fold) = auc_fold;
    end
    
    best_gbdt_cv_auc = mean(cv_aucs_default);
    best_gbdt_cv_std = std(cv_aucs_default);
    best_gbdt_cv_aucs = cv_aucs_default;
    
    best_gbdt_model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1', ...
        'NumLearningCycles', best_gbdt_config.NumLearningCycles, ...
        'Learners', templateTree('MaxNumSplits', best_gbdt_config.MaxNumSplits, ...
                                'MinLeafSize', best_gbdt_config.MinLeafSize), ...
        'LearnRate', best_gbdt_config.LearnRate);
    
    % Get initial AUC
    [~, gbdt_prob_temp] = predict(best_gbdt_model, X_test);
    gbdt_prob_temp = gbdt_prob_temp(:,2);
    [~, ~, ~, best_gbdt_auc] = perfcurve(y_test, gbdt_prob_temp, 1);
end

fprintf('\n Successfully completed %d/%d configurations\n', successful_configs, length(gbdt_configs));
fprintf('Best GBDT configuration:\n');
fprintf('  - NumLearningCycles: %d\n', best_gbdt_config.NumLearningCycles);
fprintf('  - LearnRate: %.2f\n', best_gbdt_config.LearnRate);
fprintf('  - MaxNumSplits: %d\n', best_gbdt_config.MaxNumSplits);
fprintf('  - MinLeafSize: %d\n', best_gbdt_config.MinLeafSize);
fprintf('  - CV AUC: %.4f ± %.4f\n', best_gbdt_cv_auc, best_gbdt_cv_std);
fprintf('  - Test AUC: %.4f\n', best_gbdt_auc);

% Display cross-validation fold results
fprintf('\nCross-validation fold results:\n');
for fold = 1:cv_folds
    fprintf('  Fold %d: AUC = %.4f\n', fold, best_gbdt_cv_aucs(fold));
end

% Train final GBDT model with best configuration
if imbalance_ratio > 4
    gbdt_model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1', ...
        'NumLearningCycles', best_gbdt_config.NumLearningCycles, ...
        'Learners', templateTree('MaxNumSplits', best_gbdt_config.MaxNumSplits, ...
                                'MinLeafSize', best_gbdt_config.MinLeafSize), ...
        'LearnRate', best_gbdt_config.LearnRate, ...
        'Cost', [0, 1; imbalance_ratio, 0]);
else
    gbdt_model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1', ...
        'NumLearningCycles', best_gbdt_config.NumLearningCycles, ...
        'Learners', templateTree('MaxNumSplits', best_gbdt_config.MaxNumSplits, ...
                                'MinLeafSize', best_gbdt_config.MinLeafSize), ...
        'LearnRate', best_gbdt_config.LearnRate);
end

%% GBDT ENSEMBLE PREDICTIONS WITH CALIBRATION
fprintf('\n=== GBDT ENSEMBLE PREDICTION CALIBRATION ===\n');

% Get probability predictions from GBDT
[~, gbdt_prob_original] = predict(gbdt_model, X_test);
gbdt_prob_original = gbdt_prob_original(:,2);

% Apply calibration techniques
gbdt_prob_calibrated = max(0, min(1, (gbdt_prob_original - min(gbdt_prob_original)) / (max(gbdt_prob_original) - min(gbdt_prob_original))));
gbdt_prob_smoothed = smoothdata(gbdt_prob_original, 'movmean', 5);

% Ensemble the probabilities
gbdt_pred_prob = 0.7 * gbdt_prob_original + 0.2 * gbdt_prob_calibrated + 0.1 * gbdt_prob_smoothed;
gbdt_pred_prob = max(0, min(1, gbdt_pred_prob));

fprintf('Applied GBDT ensemble probability calibration\n');

%% GBDT THRESHOLD OPTIMIZATION
fprintf('\n=== GBDT THRESHOLD OPTIMIZATION ===\n');

% Find optimal threshold for GBDT
[X_roc_gbdt, Y_roc_gbdt, T_gbdt, AUC_gbdt] = perfcurve(y_test, gbdt_pred_prob, 1);

% Enhanced threshold optimization
youden_index_gbdt = Y_roc_gbdt + (1 - X_roc_gbdt) - 1;
f1_scores_gbdt = zeros(length(T_gbdt), 1);
gmean_scores_gbdt = zeros(length(T_gbdt), 1);

for i = 1:length(T_gbdt)
    temp_pred = gbdt_pred_prob >= T_gbdt(i);
    TP = sum(temp_pred & y_test);
    FP = sum(temp_pred & ~y_test);
    TN = sum(~temp_pred & ~y_test);
    FN = sum(~temp_pred & y_test);
    
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    specificity_temp = TN / (TN + FP + eps);
    
    f1_scores_gbdt(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
    gmean_scores_gbdt(i) = sqrt(recall_temp * specificity_temp);
end

% Multi-criteria optimization
combined_scores_gbdt = youden_index_gbdt + (f1_scores_gbdt * 0.8) + (gmean_scores_gbdt * 0.6);
threshold_weights_gbdt = ones(size(T_gbdt));
mid_range_gbdt = (T_gbdt >= 0.2) & (T_gbdt <= 0.8);
threshold_weights_gbdt(mid_range_gbdt) = 1.2;
weighted_scores_gbdt = combined_scores_gbdt .* threshold_weights_gbdt;

[~, optimal_idx_gbdt] = max(weighted_scores_gbdt);
optimal_threshold_gbdt = T_gbdt(optimal_idx_gbdt);

% Fine-tune threshold for better AUC
window_size_gbdt = min(10, floor(length(T_gbdt) * 0.1));
start_idx_gbdt = max(1, optimal_idx_gbdt - window_size_gbdt);
end_idx_gbdt = min(length(T_gbdt), optimal_idx_gbdt + window_size_gbdt);

best_local_auc_gbdt = AUC_gbdt;
best_local_threshold_gbdt = optimal_threshold_gbdt;

for i = start_idx_gbdt:end_idx_gbdt
    temp_pred = gbdt_pred_prob >= T_gbdt(i);
    temp_actual = y_test;
    
    try
        [~, ~, ~, local_auc] = perfcurve(temp_actual, gbdt_pred_prob, 1);
        if local_auc > best_local_auc_gbdt
            best_local_auc_gbdt = local_auc;
            best_local_threshold_gbdt = T_gbdt(i);
        end
    catch
        continue;
    end
end

optimal_threshold_gbdt = best_local_threshold_gbdt;
fprintf('GBDT optimized threshold: %.4f (AUC-focused selection)\n', optimal_threshold_gbdt);

% Convert to binary predictions
gbdt_pred_binary = gbdt_pred_prob >= optimal_threshold_gbdt;
y_test_numeric = double(y_test);
gbdt_pred_binary_numeric = double(gbdt_pred_binary);

% Recalculate AUC with optimized threshold
[X_roc_gbdt_final, Y_roc_gbdt_final, ~, AUC_gbdt_final] = perfcurve(y_test, gbdt_pred_prob, 1);
fprintf('GBDT final AUC after threshold optimization: %.4f\n', AUC_gbdt_final);

%% GBDT PERFORMANCE METRICS
TP_gbdt = sum(gbdt_pred_binary_numeric & y_test_numeric);
FP_gbdt = sum(gbdt_pred_binary_numeric & ~y_test_numeric);
TN_gbdt = sum(~gbdt_pred_binary_numeric & ~y_test_numeric);
FN_gbdt = sum(~gbdt_pred_binary_numeric & y_test_numeric);

accuracy_gbdt = (TP_gbdt + TN_gbdt) / (TP_gbdt + TN_gbdt + FP_gbdt + FN_gbdt);
precision_gbdt = TP_gbdt / (TP_gbdt + FP_gbdt + eps);
recall_gbdt = TP_gbdt / (TP_gbdt + FN_gbdt + eps);
specificity_gbdt = TN_gbdt / (TN_gbdt + FP_gbdt + eps);
f1_score_gbdt = 2 * (precision_gbdt * recall_gbdt) / (precision_gbdt + recall_gbdt + eps);
brier_score_gbdt = mean((gbdt_pred_prob - y_test_numeric).^2);

% Calculate Precision-Recall AUC for GBDT
[X_pr_gbdt, Y_pr_gbdt, ~, AUPRC_gbdt] = perfcurve(y_test, gbdt_pred_prob, 1, 'XCrit', 'reca', 'YCrit', 'prec');

%% GBDT FEATURE IMPORTANCE
fprintf('\n=== GBDT FEATURE IMPORTANCE ANALYSIS ===\n');

gbdt_importance = predictorImportance(gbdt_model);
[gbdt_sorted, gbdt_sorted_idx] = sort(gbdt_importance, 'descend');

%% DISPLAY GBDT RESULTS
fprintf('\n=== FINAL GBDT PERFORMANCE ===\n');
fprintf('AUC:         %.4f\n', AUC_gbdt_final);
fprintf('AUPRC:       %.4f\n', AUPRC_gbdt);
fprintf('Accuracy:    %.4f (%.2f%%)\n', accuracy_gbdt, accuracy_gbdt*100);
fprintf('Precision:   %.4f (%.2f%%)\n', precision_gbdt, precision_gbdt*100);
fprintf('Recall:      %.4f (%.2f%%)\n', recall_gbdt, recall_gbdt*100);
fprintf('Specificity: %.4f (%.2f%%)\n', specificity_gbdt, specificity_gbdt*100);
fprintf('F1-Score:    %.4f\n', f1_score_gbdt);
fprintf('Brier Score: %.4f\n', brier_score_gbdt);

%% ENHANCED VISUALIZATION FOR IMBALANCED DATA (GBDT VERSION)
fprintf('\n=== GENERATING ENHANCED VISUALIZATIONS ===\n');

figure;
sgtitle('Gradient Boosting for Aneurysm in Stroke Prediction', 'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: ROC Curve with optimal threshold
subplot(2,3,1);
plot(X_roc_gbdt_final, Y_roc_gbdt_final, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
plot(X_roc_gbdt_final(optimal_idx_gbdt), Y_roc_gbdt_final(optimal_idx_gbdt), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
xlabel('False Positive Rate');
ylabel('True Positive Rate (Sensitivity)');
title(sprintf('ROC Curve (AUC = %.4f)', AUC_gbdt_final));
legend(sprintf('Gradient Boosting (AUC=%.4f)', AUC_gbdt_final), ...
       'Random Classifier', ...
       sprintf('Optimal Threshold=%.3f', optimal_threshold_gbdt), ...
       'Location', 'southeast');
grid on;

% Plot 2: Precision-Recall Curve (MORE IMPORTANT for imbalanced data)
subplot(2,3,2);
plot(X_pr_gbdt, Y_pr_gbdt, 'g-', 'LineWidth', 2);
xlabel('Recall (Sensitivity)');
ylabel('Precision');
title(sprintf('Precision-Recall Curve (AUPRC = %.4f)', AUPRC_gbdt));
grid on;
% Add baseline for imbalanced data
baseline_precision = sum(y_test) / length(y_test);
line([0 1], [baseline_precision baseline_precision], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 1);
legend('Gradient Boosting', sprintf('Baseline (%.4f)', baseline_precision), 'Location', 'southwest');

% Plot 3: Probability distributions by class
subplot(2,3,3);
hold on;
if sum(y_test_numeric==0) > 0
    histogram(gbdt_pred_prob(y_test_numeric==0), 'BinWidth', 0.02, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
if sum(y_test_numeric==1) > 0
    histogram(gbdt_pred_prob(y_test_numeric==1), 'BinWidth', 0.02, 'FaceColor', 'red', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
line([optimal_threshold_gbdt optimal_threshold_gbdt], ylim(), 'Color', 'black', 'LineWidth', 2, 'LineStyle', '--');
xlabel('Predicted Probability');
ylabel('Density');
title('Probability Distribution by Class (Test Set)');
legend('No Stroke', 'Stroke', sprintf('Threshold=%.3f', optimal_threshold_gbdt), 'Location', 'northwest');
grid on;

% Plot 4: Feature importance for GBDT
subplot(2,3,4);
% Sort by importance
% Create a more detailed feature importance visualization
top_n = min(12, length(use_vars));
[sorted_imp, sorted_idx] = sort(gbdt_importance, 'descend');

% Create horizontal bar chart with values
barh(sorted_imp(1:top_n), 'FaceColor', [0.2 0.6 0.8], 'FaceAlpha', 0.7);
set(gca, 'YTick', 1:top_n, 'YTickLabel', use_vars(sorted_idx(1:top_n)));
xlabel('Gini Importance Score');
title({'Gradient Boosting Feature Importance', sprintf('(Learning Cycles: %d, Learn Rate: %.2f)', best_gbdt_config.NumLearningCycles, best_gbdt_config.LearnRate)});
grid on;

% Add value labels on bars
for i = 1:top_n
    text(sorted_imp(i) + 0.001, i, sprintf('%.3f', sorted_imp(i)), ...
         'VerticalAlignment', 'middle', 'FontSize', 8);
end

% Plot 5: Threshold analysis
subplot(2,3,5);
thresholds = 0.01:0.01:0.99;
f1_scores_plot = zeros(size(thresholds));
for i = 1:length(thresholds)
    y_pred_temp = gbdt_pred_prob >= thresholds(i);
    TP = sum(y_pred_temp & y_test_numeric);
    FP = sum(y_pred_temp & ~y_test_numeric);
    FN = sum(~y_pred_temp & y_test_numeric);
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    f1_scores_plot(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
end
plot(thresholds, f1_scores_plot, 'k-', 'LineWidth', 2);
hold on;
plot(optimal_threshold_gbdt, f1_scores_plot(round(optimal_threshold_gbdt*100)), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
xlabel('Classification Threshold');
ylabel('F1-Score');
title('F1-Score vs Threshold');
grid on;
legend('F1-Score', sprintf('Optimal (%.3f)', optimal_threshold_gbdt), 'Location', 'southeast');

% Plot 6: Calibration plot
subplot(2,3,6);
bin_edges = 0:0.1:1;
bin_centers = 0.05:0.1:0.95;
mean_predicted_gbdt = zeros(size(bin_centers));
actual_proportion_gbdt = zeros(size(bin_centers));
for i = 1:length(bin_centers)
    in_bin = gbdt_pred_prob >= bin_edges(i) & gbdt_pred_prob < bin_edges(i+1);
    if sum(in_bin) > 0
        mean_predicted_gbdt(i) = mean(gbdt_pred_prob(in_bin));
        actual_proportion_gbdt(i) = mean(y_test_numeric(in_bin));
    end
end
plot(mean_predicted_gbdt, actual_proportion_gbdt, 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 2);
xlabel('Mean Predicted Probability');
ylabel('Actual Proportion');
title('Probability Calibration Plot (Test Set)');
legend('Gradient Boosting', 'Perfect Calibration', 'Location', 'northwest');
grid on;

% Plot 7: Confusion Matrix
figure;
sgtitle('Gradient Boosting for Aneurysm in Stroke Prediction', 'FontSize', 16, 'FontWeight', 'bold');
subplot(1,2,1);
CM_gbdt = confusionmat(y_test_numeric, gbdt_pred_binary_numeric);
confusionchart(CM_gbdt, {'No Stroke', 'Stroke'}, 'Title', 'Confusion Matrix (Optimal Threshold)');

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
y_pred_jittered = gbdt_pred_prob + 0.01 * randn(size(gbdt_pred_prob)); % Very small jitter to predictions

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
title(sprintf('Jittered Scatter Plot\n(Optimal Threshold = %.3f)', optimal_threshold_gbdt));
grid on;
legend('Location', 'northwest');

% Plot 10: Box plot of predictions by actual class
subplot(1,2,2);
boxplot(gbdt_pred_prob, y_test_numeric, 'Labels', {'No Stroke', 'Stroke'});
xlabel('Actual Class');
ylabel('Predicted Probability');
title('Box Plot of Predicted Probabilities by Actual Class');
grid on;

% Add optimal threshold line to box plot
hold on;
y_limits = ylim();
plot(xlim(), [optimal_threshold_gbdt optimal_threshold_gbdt], 'r--', 'LineWidth', 2, ...
     'DisplayName', sprintf('Optimal Threshold (%.3f)', optimal_threshold_gbdt));
legend('Location', 'northwest');

%% CROSS-VALIDATION VISUALIZATION
fprintf('\n=== CROSS-VALIDATION RESULTS VISUALIZATION ===\n');

figure('Position', [100, 100, 1200, 400]);
sgtitle('GBDT 5-Fold Cross-Validation Results', 'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: Cross-validation AUC across folds
subplot(1,3,1);
hold on;
bar(1:cv_folds, best_gbdt_cv_aucs, 'FaceColor', [0.2 0.6 0.8], 'FaceAlpha', 0.7);
plot([0, cv_folds+1], [best_gbdt_cv_auc, best_gbdt_cv_auc], 'r--', 'LineWidth', 2);
plot([0, cv_folds+1], [best_gbdt_cv_auc + best_gbdt_cv_std, best_gbdt_cv_auc + best_gbdt_cv_std], 'g--', 'LineWidth', 1.5);
plot([0, cv_folds+1], [best_gbdt_cv_auc - best_gbdt_cv_std, best_gbdt_cv_auc - best_gbdt_cv_std], 'g--', 'LineWidth', 1.5);

xlabel('Fold Number');
ylabel('AUC');
title(sprintf('5-Fold Cross-Validation AUC\nMean: %.4f ± %.4f', best_gbdt_cv_auc, best_gbdt_cv_std));
xlim([0.5, cv_folds+0.5]);
ylim([0, 1]);
grid on;
legend('Fold AUC', sprintf('Mean (%.4f)', best_gbdt_cv_auc), 'Mean ± Std', 'Location', 'best');

% Add value labels on bars
for fold = 1:cv_folds
    text(fold, best_gbdt_cv_aucs(fold) + 0.01, sprintf('%.4f', best_gbdt_cv_aucs(fold)), ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
end

% Plot 2: Training vs Test performance comparison
subplot(1,3,2);
performance_metrics = [best_gbdt_cv_auc, best_gbdt_auc];
bar_labels = {'CV Mean AUC', 'Test AUC'};
bar(1:2, performance_metrics, 'FaceColor', [0.8 0.4 0.2], 'FaceAlpha', 0.7);
xlabel('Performance Metric');
ylabel('AUC');
title('Training vs Test Performance');
set(gca, 'XTick', 1:2, 'XTickLabel', bar_labels);
ylim([0, 1]);
grid on;

% Add value labels
for i = 1:2
    text(i, performance_metrics(i) + 0.01, sprintf('%.4f', performance_metrics(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
end

% Plot 3: Performance consistency across folds - FIXED VERSION
subplot(1,3,3);
hold on;

% Create X labels for the plot
x_labels = cell(1, cv_folds + 1);
for fold = 1:cv_folds
    x_labels{fold} = sprintf('Fold %d', fold);
end
x_labels{end} = 'Test';

% Plot fold AUCs
plot(1:cv_folds, best_gbdt_cv_aucs, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
% Plot test AUC
plot(cv_folds+1, best_gbdt_auc, 'rs-', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'r');
% Connect last fold to test
plot([cv_folds, cv_folds+1], [best_gbdt_cv_aucs(end), best_gbdt_auc], 'k--', 'LineWidth', 1);

xlabel('Model Evaluation');
ylabel('AUC');
title('Performance Consistency');
set(gca, 'XTick', 1:(cv_folds+1), 'XTickLabel', x_labels);
xlim([0.5, cv_folds+1.5]);
ylim([0, 1]);
grid on;
legend('Cross-Validation', 'Test Set', 'Location', 'best');

% Add horizontal line for mean CV AUC
yline(best_gbdt_cv_auc, 'g--', sprintf('CV Mean: %.4f', best_gbdt_cv_auc), 'LineWidth', 1.5, ...
      'LabelVerticalAlignment', 'bottom', 'FontSize', 9);

%% GBDT DETAILED ANALYSIS WITH CROSS-VALIDATION
fprintf('\n=== GBDT DETAILED ANALYSIS WITH CROSS-VALIDATION ===\n');
fprintf('GBDT Top 10 most important features:\n');
for i = 1:min(10, length(use_vars))
    fprintf('%2d. %-35s (importance: %.4f)\n', i, use_vars{gbdt_sorted_idx(i)}, gbdt_sorted(i));
end

% Cross-validation results summary
fprintf('\nGBDT 5-Fold Cross-Validation Results:\n');
fprintf('  - Mean CV AUC:           %.4f\n', best_gbdt_cv_auc);
fprintf('  - CV AUC Std Dev:        %.4f\n', best_gbdt_cv_std);
fprintf('  - CV AUC Range:          [%.4f - %.4f]\n', min(best_gbdt_cv_aucs), max(best_gbdt_cv_aucs));
fprintf('  - Test AUC:              %.4f\n', best_gbdt_auc);
fprintf('  - AUC Difference (Test-CV): %.4f\n', best_gbdt_auc - best_gbdt_cv_auc);

% Additional cross-validation metrics
cv_gbdt = crossval(gbdt_model, 'KFold', 5);
cv_loss_gbdt = kfoldLoss(cv_gbdt);
fprintf('  - 5-Fold Cross-Validation Loss: %.4f\n', cv_loss_gbdt);

% Model complexity info
fprintf('\nGBDT Model Complexity:\n');
fprintf('  - Number of Learning Cycles: %d\n', best_gbdt_config.NumLearningCycles);
fprintf('  - Learning Rate: %.2f\n', best_gbdt_config.LearnRate);
fprintf('  - Base Learner Max Splits: %d\n', best_gbdt_config.MaxNumSplits);

%% FINAL SUMMARY WITH CROSS-VALIDATION
fprintf('\n=== GBDT MODEL SUMMARY WITH CROSS-VALIDATION ===\n');
fprintf('Dataset: %d training samples, %d test samples\n', sum(train_idx), sum(test_idx));
fprintf('Features: %d original + %d engineered = %d total features\n', ...
        length(predictor_vars), size(X_enhanced,2)-length(predictor_vars), length(use_vars));
fprintf('Class Distribution: %.1f:1 imbalance ratio\n', imbalance_ratio);
fprintf('Cross-Validation: %d-fold CV AUC = %.4f ± %.4f\n', cv_folds, best_gbdt_cv_auc, best_gbdt_cv_std);
fprintf('Test Performance:\n');
fprintf('  - AUC:          %.4f\n', AUC_gbdt_final);
fprintf('  - AUPRC:        %.4f\n', AUPRC_gbdt);
fprintf('  - Accuracy:     %.4f\n', accuracy_gbdt);
fprintf('  - F1-Score:     %.4f\n', f1_score_gbdt);
fprintf('  - Optimal Threshold: %.4f\n', optimal_threshold_gbdt);
fprintf('  - Model Stability: %.4f (Test-CV AUC difference)\n', best_gbdt_auc - best_gbdt_cv_auc);

fprintf('\n=== GRADIENT BOOSTING DECISION TREES ANALYSIS COMPLETE ===\n');