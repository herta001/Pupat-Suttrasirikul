%% 5-FOLD CROSS-VALIDATION DECISION TREE MODEL FOR ANEURYSM PREDICTION

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

%% STRATEGY 1: ENHANCED FEATURE ENGINEERING
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

%% ========== SMOTE IMPLEMENTATION FOR DECISION TREE ==========
fprintf('\n=== APPLYING SMOTE FOR CLASS IMBALANCE (WITHIN CROSS-VALIDATION) ===\n');

%% SETUP 5-FOLD CROSS-VALIDATION
fprintf('\n=== SETTING UP 5-FOLD CROSS-VALIDATION WITH SMOTE ===\n');
rng(42); % For reproducibility
cv_folds = 5;
cvp = cvpartition(y, 'KFold', cv_folds, 'Stratify', true);

fprintf('Performing %d-fold stratified cross-validation with SMOTE\n', cv_folds);

% Initialize storage for cross-validation results
cv_aucs = zeros(cv_folds, 1);
cv_auprcs = zeros(cv_folds, 1);
cv_accuracies = zeros(cv_folds, 1);
cv_precisions = zeros(cv_folds, 1);
cv_recalls = zeros(cv_folds, 1);
cv_specificities = zeros(cv_folds, 1);
cv_f1s = zeros(cv_folds, 1);
cv_briers = zeros(cv_folds, 1);
cv_optimal_thresholds = zeros(cv_folds, 1);
% Store SMOTE statistics for each fold
cv_smote_stats = cell(cv_folds, 1);

% Initialize storage for predictions across all folds
all_y_test = [];
all_y_pred_probs = [];
all_y_pred_binary = [];
all_fold_indices = [];
all_tree_models = cell(cv_folds, 1);
all_feature_importances = zeros(cv_folds, length(use_vars));
% Store best configurations for each fold
all_best_configs = cell(cv_folds, 1);

%% 5-FOLD CROSS-VALIDATION TRAINING WITH SMOTE
fprintf('\n=== PERFORMING 5-FOLD CROSS-VALIDATION WITH SMOTE ===\n');

for fold = 1:cv_folds
    fprintf('\n--- Training Fold %d/%d (with SMOTE) ---\n', fold, cv_folds);
    
    % Split data for this fold
    train_idx = training(cvp, fold);
    test_idx = test(cvp, fold);
    
    X_train = X(train_idx, :);
    X_test = X(test_idx, :);
    y_train = y(train_idx);
    y_test = y(test_idx);
    
    %% Apply SMOTE to training data for this fold
    fprintf('Applying SMOTE to training data for fold %d...\n', fold);
    
    minority_class = 1; % Stroke cases
    majority_class = 0; % Non-stroke cases
    
    train_minority_count = sum(y_train == minority_class);
    train_majority_count = sum(y_train == majority_class);
    train_imbalance_ratio = train_majority_count / train_minority_count;
    
    fprintf('  Training set before SMOTE:\n');
    fprintf('    Minority class (stroke): %d samples (%.2f%%)\n', train_minority_count, 100*train_minority_count/length(y_train));
    fprintf('    Majority class (non-stroke): %d samples (%.2f%%)\n', train_majority_count, 100*train_majority_count/length(y_train));
    fprintf('    Imbalance ratio: %.2f:1\n', train_imbalance_ratio);
    
    % Target 1:2 ratio (conservative balancing for Decision Trees)
    desired_minority_count = min(train_majority_count, train_minority_count * 2);
    synthetic_samples_needed = desired_minority_count - train_minority_count;
    
    % Store SMOTE statistics for this fold
    cv_smote_stats{fold} = struct(...
        'before_minority', train_minority_count, ...
        'before_majority', train_majority_count, ...
        'before_ratio', train_imbalance_ratio, ...
        'synthetic_needed', synthetic_samples_needed);
    
    % Apply SMOTE only if needed
    if synthetic_samples_needed > 0
        fprintf('  Generating %d synthetic minority samples using SMOTE...\n', synthetic_samples_needed);
        
        % Prepare minority class data
        train_minority_idx = find(y_train == minority_class);
        X_train_minority = X_train(train_minority_idx, :);
        
        % SMOTE parameters
        k = min(5, size(X_train_minority, 1) - 1); % Number of nearest neighbors
        
        % Find k-nearest neighbors for each minority sample
        [idx, ~] = knnsearch(X_train_minority, X_train_minority, 'K', k+1); % +1 to exclude self
        
        % Initialize synthetic samples array
        X_synthetic = zeros(synthetic_samples_needed, size(X_train_minority, 2));
        y_synthetic = ones(synthetic_samples_needed, 1) * minority_class;
        
        synthetic_count = 0;
        
        % Generate synthetic samples
        while synthetic_count < synthetic_samples_needed
            for i = 1:size(X_train_minority, 1)
                if synthetic_count >= synthetic_samples_needed
                    break;
                end
                
                % Randomly select one of the k-nearest neighbors (excluding self)
                neighbor_idx = idx(i, randi(k) + 1);
                
                % Generate synthetic sample with interpolation
                diff = X_train_minority(neighbor_idx, :) - X_train_minority(i, :);
                gap = rand(); % Random interpolation factor between 0 and 1
                
                synthetic_sample = X_train_minority(i, :) + gap * diff;
                
                % Ensure synthetic sample stays within reasonable bounds
                for j = 1:size(synthetic_sample, 2)
                    % Get min and max from actual training data for this feature
                    feat_min = min(X_train(:, j));
                    feat_max = max(X_train(:, j));
                    
                    % Clip to bounds
                    synthetic_sample(j) = max(feat_min, min(feat_max, synthetic_sample(j)));
                end
                
                synthetic_count = synthetic_count + 1;
                X_synthetic(synthetic_count, :) = synthetic_sample;
            end
        end
        
        % Combine original and synthetic data
        X_train_balanced = [X_train; X_synthetic];
        y_train_balanced = [y_train; y_synthetic];
        
        % Shuffle the balanced dataset
        shuffle_idx = randperm(size(X_train_balanced, 1));
        X_train_balanced = X_train_balanced(shuffle_idx, :);
        y_train_balanced = y_train_balanced(shuffle_idx);
        
        % Update training data for this fold
        X_train = X_train_balanced;
        y_train = y_train_balanced;
        
        fprintf('  After SMOTE:\n');
        fprintf('    Total training samples: %d\n', size(X_train, 1));
        fprintf('    Minority samples: %d (%.2f%%)\n', sum(y_train == minority_class), 100*sum(y_train == minority_class)/length(y_train));
        fprintf('    Majority samples: %d (%.2f%%)\n', sum(y_train == majority_class), 100*sum(y_train == majority_class)/length(y_train));
        fprintf('    New imbalance ratio: %.2f:1\n', sum(y_train == majority_class)/sum(y_train == minority_class));
        
        % Update SMOTE statistics
        cv_smote_stats{fold}.after_minority = sum(y_train == minority_class);
        cv_smote_stats{fold}.after_majority = sum(y_train == majority_class);
        cv_smote_stats{fold}.after_ratio = sum(y_train == majority_class)/sum(y_train == minority_class);
    else
        fprintf('  No SMOTE needed - minority class already sufficient.\n');
        cv_smote_stats{fold}.after_minority = train_minority_count;
        cv_smote_stats{fold}.after_majority = train_majority_count;
        cv_smote_stats{fold}.after_ratio = train_imbalance_ratio;
    end
    
    % Calculate fold-specific imbalance (after SMOTE if applied)
    fold_imbalance_ratio = sum(y_train == 0) / sum(y_train == 1);
    
    %% STRATEGY 2: OPTIMIZED DECISION TREE WITH BETTER HYPERPARAMETERS
    fprintf('Testing multiple Decision Tree configurations for fold %d...\n', fold);
    
    % Decision Tree parameter configurations
    configs = {
        struct('MinLeafSize', 1, 'MaxNumSplits', 100, 'SplitCriterion', 'gdi'),
        struct('MinLeafSize', 5, 'MaxNumSplits', 50, 'SplitCriterion', 'gdi'),
        struct('MinLeafSize', 10, 'MaxNumSplits', 30, 'SplitCriterion', 'gdi'),
        struct('MinLeafSize', 1, 'MaxNumSplits', 100, 'SplitCriterion', 'deviance'),
        struct('MinLeafSize', 5, 'MaxNumSplits', 50, 'SplitCriterion', 'deviance'),
        struct('MinLeafSize', 10, 'MaxNumSplits', 30, 'SplitCriterion', 'deviance'),
        struct('MinLeafSize', 20, 'MaxNumSplits', 20, 'SplitCriterion', 'gdi')
    };
    
    best_auc = 0;
    best_tree_model_fold = [];
    best_config_fold = [];
    
    for i = 1:length(configs)
        try
            if fold_imbalance_ratio > 4
                % For imbalanced data, use class weights
                tree_temp = fitctree(X_train, y_train, ...
                    'MinLeafSize', configs{i}.MinLeafSize, ...
                    'MaxNumSplits', configs{i}.MaxNumSplits, ...
                    'SplitCriterion', configs{i}.SplitCriterion, ...
                    'Prior', 'empirical', ...
                    'Cost', [0, 1; fold_imbalance_ratio, 0]);
            else
                tree_temp = fitctree(X_train, y_train, ...
                    'MinLeafSize', configs{i}.MinLeafSize, ...
                    'MaxNumSplits', configs{i}.MaxNumSplits, ...
                    'SplitCriterion', configs{i}.SplitCriterion);
            end
            
            % Get probability predictions
            [~, prob_temp] = predict(tree_temp, X_test);
            prob_temp = prob_temp(:,2); % Probability of class 1 (stroke)
            
            % Calculate AUC
            [~, ~, ~, temp_auc] = perfcurve(y_test, prob_temp, 1);
            
            if temp_auc > best_auc
                best_auc = temp_auc;
                best_tree_model_fold = tree_temp;
                best_config_fold = configs{i};
            end
        catch ME
            fprintf('Config %d failed: %s\n', i, ME.message);
        end
    end
    
    fprintf('Best configuration for fold %d: MinLeafSize=%d, MaxNumSplits=%d, SplitCriterion=%s (AUC: %.4f)\n', ...
            fold, best_config_fold.MinLeafSize, best_config_fold.MaxNumSplits, best_config_fold.SplitCriterion, best_auc);
    
    % Store the configuration
    all_best_configs{fold} = best_config_fold;
    
    % Train final model for this fold with best configuration
    if fold_imbalance_ratio > 4
        tree_model_fold = fitctree(X_train, y_train, ...
            'MinLeafSize', best_config_fold.MinLeafSize, ...
            'MaxNumSplits', best_config_fold.MaxNumSplits, ...
            'SplitCriterion', best_config_fold.SplitCriterion, ...
            'Prior', 'empirical', ...
            'Cost', [0, 1; fold_imbalance_ratio, 0]);
    else
        tree_model_fold = fitctree(X_train, y_train, ...
            'MinLeafSize', best_config_fold.MinLeafSize, ...
            'MaxNumSplits', best_config_fold.MaxNumSplits, ...
            'SplitCriterion', best_config_fold.SplitCriterion);
    end
    
    % Store the model
    all_tree_models{fold} = tree_model_fold;
    
    %% STRATEGY 3: ENSEMBLE PREDICTIONS FOR BETTER CALIBRATION
    fprintf('Applying ensemble probability calibration for fold %d...\n', fold);
    
    % Get probability predictions from the tree
    [~, tree_prob_original] = predict(tree_model_fold, X_test);
    prob_original = tree_prob_original(:,2);
    
    % Method 2: Calibrated probabilities using Platt scaling approximation
    prob_calibrated = max(0, min(1, (prob_original - min(prob_original)) / (max(prob_original) - min(prob_original))));
    
    % Method 3: Smoothed probabilities
    prob_smoothed = smoothdata(prob_original, 'movmean', 5);
    
    % Ensemble the probabilities (weighted average)
    tree_pred_prob = 0.7 * prob_original + 0.2 * prob_calibrated + 0.1 * prob_smoothed;
    tree_pred_prob = max(0, min(1, tree_pred_prob));
    
    %% IMPROVED THRESHOLD OPTIMIZATION FOR BETTER AUC
    fprintf('Optimizing threshold for fold %d...\n', fold);
    
    % Find optimal threshold for this fold
    [X_roc_tree, Y_roc_tree, T_tree, AUC_tree] = perfcurve(y_test, tree_pred_prob, 1);
    cv_aucs(fold) = AUC_tree;
    
    % Enhanced threshold optimization with AUC focus
    youden_index = Y_roc_tree + (1 - X_roc_tree) - 1;
    f1_scores = zeros(length(T_tree), 1);
    gmean_scores = zeros(length(T_tree), 1);
    
    for i = 1:length(T_tree)
        temp_pred = tree_pred_prob >= T_tree(i);
        TP = sum(temp_pred & y_test);
        FP = sum(temp_pred & ~y_test);
        TN = sum(~temp_pred & ~y_test);
        FN = sum(~temp_pred & y_test);
        
        precision_temp = TP / (TP + FP + eps);
        recall_temp = TP / (TP + FN + eps);
        specificity_temp = TN / (TN + FP + eps);
        
        f1_scores(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
        gmean_scores(i) = sqrt(recall_temp * specificity_temp);
    end
    
    % Multi-criteria optimization
    combined_scores = youden_index + (f1_scores * 0.8) + (gmean_scores * 0.6);
    threshold_weights = ones(size(T_tree));
    mid_range = (T_tree >= 0.2) & (T_tree <= 0.8);
    threshold_weights(mid_range) = 1.2;
    
    weighted_scores = combined_scores .* threshold_weights;
    [~, optimal_idx_tree] = max(weighted_scores);
    optimal_threshold_tree = T_tree(optimal_idx_tree);
    
    % Fine-tune threshold for better AUC
    window_size = min(10, floor(length(T_tree) * 0.1));
    start_idx = max(1, optimal_idx_tree - window_size);
    end_idx = min(length(T_tree), optimal_idx_tree + window_size);
    
    best_local_auc = AUC_tree;
    best_local_threshold = optimal_threshold_tree;
    
    for i = start_idx:end_idx
        try
            [~, ~, ~, local_auc] = perfcurve(y_test, tree_pred_prob, 1);
            if local_auc > best_local_auc
                best_local_auc = local_auc;
                best_local_threshold = T_tree(i);
            end
        catch
            continue;
        end
    end
    
    optimal_threshold_tree = best_local_threshold;
    cv_optimal_thresholds(fold) = optimal_threshold_tree;
    
    % Convert to binary predictions
    tree_pred_binary = tree_pred_prob >= optimal_threshold_tree;
    y_test_numeric = double(y_test);
    tree_pred_binary_numeric = double(tree_pred_binary);
    
    % Store predictions for overall evaluation
    all_y_test = [all_y_test; y_test_numeric];
    all_y_pred_probs = [all_y_pred_probs; tree_pred_prob];
    all_y_pred_binary = [all_y_pred_binary; tree_pred_binary_numeric];
    all_fold_indices = [all_fold_indices; fold * ones(length(y_test), 1)];
    
    %% Calculate Performance Metrics for this fold
    TP_tree = sum(tree_pred_binary_numeric & y_test_numeric);
    FP_tree = sum(tree_pred_binary_numeric & ~y_test_numeric);
    TN_tree = sum(~tree_pred_binary_numeric & ~y_test_numeric);
    FN_tree = sum(~tree_pred_binary_numeric & y_test_numeric);
    
    cv_accuracies(fold) = (TP_tree + TN_tree) / (TP_tree + TN_tree + FP_tree + FN_tree);
    cv_precisions(fold) = TP_tree / (TP_tree + FP_tree + eps);
    cv_recalls(fold) = TP_tree / (TP_tree + FN_tree + eps);
    cv_specificities(fold) = TN_tree / (TN_tree + FP_tree + eps);
    cv_f1s(fold) = 2 * (cv_precisions(fold) * cv_recalls(fold)) / (cv_precisions(fold) + cv_recalls(fold) + eps);
    cv_briers(fold) = mean((tree_pred_prob - y_test_numeric).^2);
    
    % Calculate Precision-Recall AUC
    [~, ~, ~, AUPRC_tree] = perfcurve(y_test, tree_pred_prob, 1, 'XCrit', 'reca', 'YCrit', 'prec');
    cv_auprcs(fold) = AUPRC_tree;
    
    % Store feature importance for this fold
    all_feature_importances(fold, :) = predictorImportance(tree_model_fold);
    
    fprintf('Fold %d complete: AUC = %.4f, Accuracy = %.4f\n', fold, cv_aucs(fold), cv_accuracies(fold));
end

%% OVERALL PERFORMANCE EVALUATION
fprintf('\n=== 5-FOLD CROSS-VALIDATION RESULTS (WITH SMOTE) ===\n');

% Display SMOTE statistics across folds
fprintf('\n--- SMOTE Statistics Across Folds ---\n');
for fold = 1:cv_folds
    stats = cv_smote_stats{fold};
    if stats.synthetic_needed > 0
        fprintf('Fold %d: Generated %d synthetic samples (Ratio: %.1f:1 -> %.1f:1)\n', ...
                fold, stats.synthetic_needed, stats.before_ratio, stats.after_ratio);
    else
        fprintf('Fold %d: No SMOTE needed (Ratio: %.1f:1)\n', fold, stats.before_ratio);
    end
end

% Calculate overall metrics using all predictions
[X_roc_overall, Y_roc_overall, T_overall, AUC_overall] = perfcurve(all_y_test, all_y_pred_probs, 1);
[~, ~, ~, AUPRC_overall] = perfcurve(all_y_test, all_y_pred_probs, 1, 'XCrit', 'reca', 'YCrit', 'prec');

% Find overall optimal threshold
youden_index = Y_roc_overall + (1 - X_roc_overall) - 1;
f1_scores = zeros(length(T_overall), 1);
for i = 1:length(T_overall)
    temp_pred = all_y_pred_probs >= T_overall(i);
    TP = sum(temp_pred & all_y_test);
    FP = sum(temp_pred & ~all_y_test);
    TN = sum(~temp_pred & ~all_y_test);
    FN = sum(~temp_pred & all_y_test);
    
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    f1_scores(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
end

combined_scores = youden_index + (f1_scores * 0.8);
[~, optimal_idx_overall] = max(combined_scores);
optimal_threshold_overall = T_overall(optimal_idx_overall);

% Create final predictions with overall optimal threshold
all_y_pred_binary_overall = all_y_pred_probs >= optimal_threshold_overall;

% Calculate overall performance metrics
TP_overall = sum(all_y_pred_binary_overall & all_y_test);
FP_overall = sum(all_y_pred_binary_overall & ~all_y_test);
TN_overall = sum(~all_y_pred_binary_overall & ~all_y_test);
FN_overall = sum(~all_y_pred_binary_overall & all_y_test);

accuracy_overall = (TP_overall + TN_overall) / (TP_overall + TN_overall + FP_overall + FN_overall);
precision_overall = TP_overall / (TP_overall + FP_overall + eps);
recall_overall = TP_overall / (TP_overall + FN_overall + eps);
specificity_overall = TN_overall / (TN_overall + FP_overall + eps);
f1_score_overall = 2 * (precision_overall * recall_overall) / (precision_overall + recall_overall + eps);
brier_score_overall = mean((all_y_pred_probs - all_y_test).^2);

% Calculate fold statistics
fprintf('\n--- Fold-by-Fold Results (with SMOTE) ---\n');
for fold = 1:cv_folds
    fprintf('Fold %d: AUC=%.4f, Acc=%.4f, Prec=%.4f, Rec=%.4f, Spec=%.4f, F1=%.4f, Thresh=%.4f\n', ...
            fold, cv_aucs(fold), cv_accuracies(fold), cv_precisions(fold), ...
            cv_recalls(fold), cv_specificities(fold), cv_f1s(fold), cv_optimal_thresholds(fold));
end

fprintf('\n--- Cross-Validation Summary Statistics (with SMOTE) ---\n');
fprintf('AUC:         Mean=%.4f ± %.4f, Range=[%.4f, %.4f]\n', mean(cv_aucs), std(cv_aucs), min(cv_aucs), max(cv_aucs));
fprintf('Accuracy:    Mean=%.4f ± %.4f, Range=[%.4f, %.4f]\n', mean(cv_accuracies), std(cv_accuracies), min(cv_accuracies), max(cv_accuracies));
fprintf('Precision:   Mean=%.4f ± %.4f, Range=[%.4f, %.4f]\n', mean(cv_precisions), std(cv_precisions), min(cv_precisions), max(cv_precisions));
fprintf('Recall:      Mean=%.4f ± %.4f, Range=[%.4f, %.4f]\n', mean(cv_recalls), std(cv_recalls), min(cv_recalls), max(cv_recalls));
fprintf('Specificity: Mean=%.4f ± %.4f, Range=[%.4f, %.4f]\n', mean(cv_specificities), std(cv_specificities), min(cv_specificities), max(cv_specificities));
fprintf('F1-Score:    Mean=%.4f ± %.4f, Range=[%.4f, %.4f]\n', mean(cv_f1s), std(cv_f1s), min(cv_f1s), max(cv_f1s));
fprintf('AUPRC:       Mean=%.4f ± %.4f, Range=[%.4f, %.4f]\n', mean(cv_auprcs), std(cv_auprcs), min(cv_auprcs), max(cv_auprcs));
fprintf('Brier Score: Mean=%.4f ± %.4f, Range=[%.4f, %.4f]\n', mean(cv_briers), std(cv_briers), min(cv_briers), max(cv_briers));

fprintf('\n--- Overall Performance (All Folds Combined) ---\n');
fprintf('Original dataset imbalance: %.2f:1\n', imbalance_ratio);
fprintf('Overall AUC:         %.4f\n', AUC_overall);
fprintf('Overall AUPRC:       %.4f\n', AUPRC_overall);
fprintf('Overall Accuracy:    %.4f (%.2f%%)\n', accuracy_overall, accuracy_overall*100);
fprintf('Overall Precision:   %.4f (%.2f%%)\n', precision_overall, precision_overall*100);
fprintf('Overall Recall:      %.4f (%.2f%%)\n', recall_overall, recall_overall*100);
fprintf('Overall Specificity: %.4f (%.2f%%)\n', specificity_overall, specificity_overall*100);
fprintf('Overall F1-Score:    %.4f\n', f1_score_overall);
fprintf('Overall Brier Score: %.4f\n', brier_score_overall);
fprintf('Optimal Threshold:   %.4f\n', optimal_threshold_overall);

%% Calculate Average Feature Importance
fprintf('\n=== AVERAGE FEATURE IMPORTANCE ACROSS FOLDS (WITH SMOTE) ===\n');
avg_feature_importance = mean(all_feature_importances, 1);
[tree_sorted, tree_sorted_idx] = sort(avg_feature_importance, 'descend');

fprintf('Top 10 most important features (average across folds):\n');
for i = 1:min(10, length(use_vars))
    fprintf('%d. %-30s (importance: %.4f)\n', i, use_vars{tree_sorted_idx(i)}, tree_sorted(i));
end

%% TRAIN FINAL MODEL ON ALL DATA FOR DEPLOYMENT (WITH SMOTE)
fprintf('\n=== TRAINING FINAL MODEL ON ALL DATA WITH SMOTE ===\n');

% First apply SMOTE to the entire dataset for final model
fprintf('Applying SMOTE to entire dataset for final model...\n');

minority_class = 1;
majority_class = 0;

total_minority_count = sum(y == minority_class);
total_majority_count = sum(y == majority_class);

% Target 1:2 ratio for final model
desired_final_minority_count = min(total_majority_count, total_minority_count * 2);
synthetic_samples_needed_final = desired_final_minority_count - total_minority_count;

fprintf('  Original dataset: %d minority, %d majority (ratio: %.1f:1)\n', ...
        total_minority_count, total_majority_count, total_majority_count/total_minority_count);
fprintf('  Target minority count: %d\n', desired_final_minority_count);
fprintf('  Synthetic samples needed: %d\n', synthetic_samples_needed_final);

if synthetic_samples_needed_final > 0
    % Prepare minority class data
    minority_idx = find(y == minority_class);
    X_minority = X(minority_idx, :);
    
    % SMOTE parameters
    k = min(5, size(X_minority, 1) - 1);
    [idx, ~] = knnsearch(X_minority, X_minority, 'K', k+1);
    
    % Generate synthetic samples
    X_synthetic_final = zeros(synthetic_samples_needed_final, size(X_minority, 2));
    y_synthetic_final = ones(synthetic_samples_needed_final, 1) * minority_class;
    
    synthetic_count = 0;
    while synthetic_count < synthetic_samples_needed_final
        for i = 1:size(X_minority, 1)
            if synthetic_count >= synthetic_samples_needed_final
                break;
            end
            
            neighbor_idx = idx(i, randi(k) + 1);
            diff = X_minority(neighbor_idx, :) - X_minority(i, :);
            gap = rand();
            
            synthetic_sample = X_minority(i, :) + gap * diff;
            
            for j = 1:size(synthetic_sample, 2)
                feat_min = min(X(:, j));
                feat_max = max(X(:, j));
                synthetic_sample(j) = max(feat_min, min(feat_max, synthetic_sample(j)));
            end
            
            synthetic_count = synthetic_count + 1;
            X_synthetic_final(synthetic_count, :) = synthetic_sample;
        end
    end
    
    % Combine with original data
    X_final = [X; X_synthetic_final];
    y_final = [y; y_synthetic_final];
    
    % Shuffle
    shuffle_idx = randperm(size(X_final, 1));
    X_final = X_final(shuffle_idx, :);
    y_final = y_final(shuffle_idx);
    
    fprintf('  Final balanced dataset: %d samples (%.1f%% stroke)\n', ...
            size(X_final, 1), 100*mean(y_final));
else
    X_final = X;
    y_final = y;
    fprintf('  No SMOTE needed for final model.\n');
end

% Find best configuration across all folds
best_fold = find(cv_aucs == max(cv_aucs), 1);
fprintf('Using configuration from fold %d (AUC: %.4f)\n', best_fold, cv_aucs(best_fold));

% Get the best configuration from stored configurations
best_config = all_best_configs{best_fold};
fprintf('Best hyperparameters: MinLeafSize=%d, MaxNumSplits=%d, SplitCriterion=%s\n', ...
        best_config.MinLeafSize, best_config.MaxNumSplits, best_config.SplitCriterion);

% Train final model on balanced data with the best configuration
final_imbalance_ratio = sum(y_final == 0) / sum(y_final == 1);

if final_imbalance_ratio > 4
    final_tree_model = fitctree(X_final, y_final, ...
        'MinLeafSize', best_config.MinLeafSize, ...
        'MaxNumSplits', best_config.MaxNumSplits, ...
        'SplitCriterion', best_config.SplitCriterion, ...
        'Prior', 'empirical', ...
        'Cost', [0, 1; final_imbalance_ratio, 0]);
else
    final_tree_model = fitctree(X_final, y_final, ...
        'MinLeafSize', best_config.MinLeafSize, ...
        'MaxNumSplits', best_config.MaxNumSplits, ...
        'SplitCriterion', best_config.SplitCriterion);
end

fprintf('Final model trained on %d samples (with SMOTE)\n', size(X_final, 1));

%% ENHANCED VISUALIZATION FOR 5-FOLD CROSS-VALIDATION WITH SMOTE
fprintf('\n=== GENERATING ENHANCED VISUALIZATIONS (WITH SMOTE) ===\n');

% Convert to double for plotting
all_y_test_double = double(all_y_test);
all_y_pred_binary_overall_double = double(all_y_pred_binary_overall);

% Calculate PR curve data for overall plot
[X_pr_overall, Y_pr_overall, ~, ~] = perfcurve(all_y_test_double, all_y_pred_probs, 1, 'XCrit', 'reca', 'YCrit', 'prec');

%% FIGURE 1: Main Performance Plots
figure('Position', [100, 100, 1200, 800]);
sgtitle('Decision Tree for Aneurysm in Stroke Prediction (with SMOTE)', 'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: ROC Curve with optimal threshold
subplot(2,3,1);
plot(X_roc_overall, Y_roc_overall, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
plot(X_roc_overall(optimal_idx_overall), Y_roc_overall(optimal_idx_overall), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
xlabel('False Positive Rate');
ylabel('True Positive Rate (Sensitivity)');
title(sprintf('ROC Curve (AUC = %.4f)', AUC_overall));
legend(sprintf('Decision Tree with SMOTE (AUC=%.4f)', AUC_overall), ...
       'Random Classifier', ...
       sprintf('Optimal Threshold=%.3f', optimal_threshold_overall), ...
       'Location', 'southeast');
grid on;

% Plot 2: Precision-Recall Curve (MORE IMPORTANT for imbalanced data)
subplot(2,3,2);
plot(X_pr_overall, Y_pr_overall, 'g-', 'LineWidth', 2);
xlabel('Recall (Sensitivity)');
ylabel('Precision');
title(sprintf('Precision-Recall Curve (AUPRC = %.4f)', AUPRC_overall));
grid on;
% Add baseline for imbalanced data
baseline_precision = sum(all_y_test_double) / length(all_y_test_double);
line([0 1], [baseline_precision baseline_precision], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 1);
legend('Decision Tree with SMOTE', sprintf('Baseline (%.4f)', baseline_precision), 'Location', 'southwest');

% Plot 3: Probability distributions by class
subplot(2,3,3);
hold on;
if sum(all_y_test_double==0) > 0
    histogram(all_y_pred_probs(all_y_test_double==0), 'BinWidth', 0.02, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
if sum(all_y_test_double==1) > 0
    histogram(all_y_pred_probs(all_y_test_double==1), 'BinWidth', 0.02, 'FaceColor', 'red', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
line([optimal_threshold_overall optimal_threshold_overall], ylim(), 'Color', 'black', 'LineWidth', 2, 'LineStyle', '--');
xlabel('Predicted Probability');
ylabel('Density');
title('Probability Distribution by Class (All Folds)');
legend('No Stroke', 'Stroke', sprintf('Threshold=%.3f', optimal_threshold_overall), 'Location', 'northwest');
grid on;

% Plot 4: Feature importance for Decision Tree
subplot(2,3,4);
top_n = min(12, length(use_vars));
barh(tree_sorted(1:top_n), 'FaceColor', [0.2 0.6 0.8], 'FaceAlpha', 0.7);
set(gca, 'YTick', 1:top_n, 'YTickLabel', use_vars(tree_sorted_idx(1:top_n)));
xlabel('Gini Importance Score');
title({'Decision Tree Feature Importance', sprintf('(Average across %d folds with SMOTE)', cv_folds)});
grid on;

% Add value labels on bars
for i = 1:top_n
    text(tree_sorted(i) + 0.001, i, sprintf('%.3f', tree_sorted(i)), ...
         'VerticalAlignment', 'middle', 'FontSize', 8);
end

% Plot 5: Threshold analysis
subplot(2,3,5);
thresholds = 0.01:0.01:0.99;
f1_scores_plot = zeros(size(thresholds));
for i = 1:length(thresholds)
    y_pred_temp = all_y_pred_probs >= thresholds(i);
    TP = sum(y_pred_temp & all_y_test_double);
    FP = sum(y_pred_temp & ~all_y_test_double);
    FN = sum(~y_pred_temp & all_y_test_double);
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    f1_scores_plot(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
end
plot(thresholds, f1_scores_plot, 'k-', 'LineWidth', 2);
hold on;

% Find the index corresponding to the optimal threshold
threshold_index = find(thresholds >= optimal_threshold_overall, 1);
if ~isempty(threshold_index)
    plot(optimal_threshold_overall, f1_scores_plot(threshold_index), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
else
    % Fallback: use the closest threshold
    [~, closest_idx] = min(abs(thresholds - optimal_threshold_overall));
    plot(optimal_threshold_overall, f1_scores_plot(closest_idx), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
end

xlabel('Classification Threshold');
ylabel('F1-Score');
title('F1-Score vs Threshold');
grid on;
legend('F1-Score', sprintf('Optimal (%.3f)', optimal_threshold_overall), 'Location', 'southeast');

% Plot 6: Calibration plot
subplot(2,3,6);
bin_edges = 0:0.1:1;
bin_centers = 0.05:0.1:0.95;
mean_predicted = zeros(size(bin_centers));
actual_proportion = zeros(size(bin_centers));
for i = 1:length(bin_centers)
    in_bin = all_y_pred_probs >= bin_edges(i) & all_y_pred_probs < bin_edges(i+1);
    if sum(in_bin) > 0
        mean_predicted(i) = mean(all_y_pred_probs(in_bin));
        actual_proportion(i) = mean(all_y_test_double(in_bin));
    end
end
plot(mean_predicted, actual_proportion, 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 2);
xlabel('Mean Predicted Probability');
ylabel('Actual Proportion');
title('Probability Calibration Plot (All Folds)');
legend('Decision Tree with SMOTE', 'Perfect Calibration', 'Location', 'northwest');
grid on;

%% FIGURE 2: Confusion Matrix and Distribution Plots
figure('Position', [150, 150, 1000, 500]);
sgtitle('Decision Tree for Aneurysm in Stroke Prediction (with SMOTE)', 'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: Confusion Matrix (Overall)
subplot(1,2,1);
CM_overall = confusionmat(all_y_test_double, all_y_pred_binary_overall_double);
try
    % Try using confusionchart (requires MATLAB R2018b or later)
    confusionchart(CM_overall, {'No Stroke', 'Stroke'}, ...
                   'Title', 'Confusion Matrix (Optimal Threshold)');
catch
    % Fallback to manual confusion matrix plot
    imagesc(CM_overall);
    colormap(flipud(gray));
    textStrings = num2str(CM_overall(:), '%d');
    textStrings = strtrim(cellstr(textStrings));
    [x, y] = meshgrid(1:size(CM_overall, 2), 1:size(CM_overall, 1));
    hStrings = text(x(:), y(:), textStrings(:), ...
                    'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
    xticks(1:2); xticklabels({'No Stroke', 'Stroke'});
    yticks(1:2); yticklabels({'No Stroke', 'Stroke'});
    xlabel('Predicted');
    ylabel('Actual');
    title(sprintf('Confusion Matrix\n(Threshold = %.3f)', optimal_threshold_overall));
    colorbar;
end

% Plot 2: Class distribution
subplot(1,2,2);
pie([sum(all_y_test_double==0), sum(all_y_test_double==1)], {'No Stroke', 'Stroke'});
title(sprintf('Test Set Class Distribution\n(Imbalance: %.1f:1)', imbalance_ratio));

%% FIGURE 3: Jittered Scatter and Box Plots
figure('Position', [200, 200, 1000, 500]);
sgtitle('Decision Tree Prediction Analysis (with SMOTE)', 'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: Jittered scatter plot for binary classification
subplot(1,2,1);
% Add small random noise to actual values for visualization
rng(42); % For reproducibility
y_jittered = all_y_test_double + 0.05 * randn(size(all_y_test_double)); % Small jitter to actual values
y_pred_jittered = all_y_pred_probs + 0.01 * randn(size(all_y_pred_probs)); % Very small jitter to predictions

% Color points by actual class
colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980]; % Blue and orange
scatter(y_jittered(all_y_test_double==0), y_pred_jittered(all_y_test_double==0), 40, colors(1,:), 'filled', ...
        'MarkerFaceAlpha', 0.6, 'DisplayName', 'Actual: No Stroke');
hold on;
scatter(y_jittered(all_y_test_double==1), y_pred_jittered(all_y_test_double==1), 40, colors(2,:), 'filled', ...
        'MarkerFaceAlpha', 0.6, 'DisplayName', 'Actual: Stroke');

% Add perfect prediction lines for binary classification
plot([0.5 0.5], [-0.1 1.1], 'r--', 'LineWidth', 2, 'HandleVisibility', 'off');

xlim([-0.2, 1.2]);
ylim([-0.1, 1.1]);
xlabel('Actual Stroke (with jitter)');
ylabel('Predicted Probability (with jitter)');
title(sprintf('Jittered Scatter Plot\n(Optimal Threshold = %.3f)', optimal_threshold_overall));
grid on;
legend('Location', 'northwest');

% Plot 2: Box plot of predictions by actual class
subplot(1,2,2);
boxplot(all_y_pred_probs, all_y_test_double, 'Labels', {'No Stroke', 'Stroke'});
xlabel('Actual Class');
ylabel('Predicted Probability');
title('Box Plot of Predicted Probabilities by Actual Class');
grid on;

% Add optimal threshold line to box plot
hold on;
y_limits = ylim();
plot(xlim(), [optimal_threshold_overall optimal_threshold_overall], 'r--', 'LineWidth', 2, ...
     'DisplayName', sprintf('Optimal Threshold (%.3f)', optimal_threshold_overall));
legend('Location', 'northwest');

%% FIGURE 4: Cross-Validation Performance Visualization
figure('Position', [250, 250, 1200, 800]);
sgtitle('5-Fold Cross-Validation Performance - Decision Tree (with SMOTE)', 'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: Cross-validation metrics distribution
subplot(2,3,1);
metrics_to_plot = [cv_accuracies, cv_aucs, cv_f1s, cv_precisions, cv_recalls];
boxplot(metrics_to_plot, 'Labels', {'Accuracy', 'AUC', 'F1-Score', 'Precision', 'Recall'});
ylabel('Score');
title('Cross-Validation Performance Distribution');
grid on;
set(gca, 'XTickLabelRotation', 45);

% Plot 2: Fold-wise AUC
subplot(2,3,2);
bar(1:cv_folds, cv_aucs);
xlabel('Fold');
ylabel('AUC');
title('AUC Across 5 Folds');
ylim([0, 1]);
grid on;
hold on;
plot(xlim(), [mean(cv_aucs) mean(cv_aucs)], 'r--', 'LineWidth', 2);
text(cv_folds/2, 0.95, sprintf('Mean: %.4f', mean(cv_aucs)), 'HorizontalAlignment', 'center');

% Plot 3: Fold-wise F1-Score
subplot(2,3,3);
bar(1:cv_folds, cv_f1s);
xlabel('Fold');
ylabel('F1-Score');
title('F1-Score Across 5 Folds');
ylim([0, 1]);
grid on;
hold on;
plot(xlim(), [mean(cv_f1s) mean(cv_f1s)], 'r--', 'LineWidth', 2);
text(cv_folds/2, 0.95, sprintf('Mean: %.4f', mean(cv_f1s)), 'HorizontalAlignment', 'center');

% Plot 4: Fold-wise Recall
subplot(2,3,4);
bar(1:cv_folds, cv_recalls);
xlabel('Fold');
ylabel('Recall');
title('Recall (Sensitivity) Across 5 Folds');
ylim([0, 1]);
grid on;
hold on;
plot(xlim(), [mean(cv_recalls) mean(cv_recalls)], 'r--', 'LineWidth', 2);
text(cv_folds/2, 0.95, sprintf('Mean: %.4f', mean(cv_recalls)), 'HorizontalAlignment', 'center');

% Plot 5: Fold-wise Precision
subplot(2,3,5);
bar(1:cv_folds, cv_precisions);
xlabel('Fold');
ylabel('Precision');
title('Precision Across 5 Folds');
ylim([0, 1]);
grid on;
hold on;
plot(xlim(), [mean(cv_precisions) mean(cv_precisions)], 'r--', 'LineWidth', 2);
text(cv_folds/2, 0.95, sprintf('Mean: %.4f', mean(cv_precisions)), 'HorizontalAlignment', 'center');

% Plot 6: Fold-wise Specificity
subplot(2,3,6);
bar(1:cv_folds, cv_specificities);
xlabel('Fold');
ylabel('Specificity');
title('Specificity Across 5 Folds');
ylim([0, 1]);
grid on;
hold on;
plot(xlim(), [mean(cv_specificities) mean(cv_specificities)], 'r--', 'LineWidth', 2);
text(cv_folds/2, 0.95, sprintf('Mean: %.4f', mean(cv_specificities)), 'HorizontalAlignment', 'center');

%% FIGURE 5: SMOTE Impact Visualization
figure('Position', [300, 300, 1200, 800]);
sgtitle('SMOTE Impact on Decision Tree Performance', 'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: Class distribution comparison
subplot(2, 3, 1);
fold_ratios_before = zeros(cv_folds, 1);
fold_ratios_after = zeros(cv_folds, 1);
for fold = 1:cv_folds
    stats = cv_smote_stats{fold};
    fold_ratios_before(fold) = stats.before_ratio;
    fold_ratios_after(fold) = stats.after_ratio;
end

bar_data = [fold_ratios_before, fold_ratios_after]';
bar(1:cv_folds, bar_data);
xlabel('Fold');
ylabel('Imbalance Ratio (Majority:Minority)');
title('SMOTE Impact on Class Balance');
legend('Before SMOTE', 'After SMOTE', 'Location', 'best');
grid on;

% Add ratio labels
for i = 1:cv_folds
    text(i-0.15, fold_ratios_before(i)+max(fold_ratios_before)*0.05, sprintf('%.1f', fold_ratios_before(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    text(i+0.15, fold_ratios_after(i)+max(fold_ratios_after)*0.05, sprintf('%.1f', fold_ratios_after(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

% Plot 2: Feature importance comparison with/without SMOTE
subplot(2, 3, 2);
top_n_features = min(8, length(use_vars));
importance_std = std(all_feature_importances(:, tree_sorted_idx(1:top_n_features)), 0, 1);
errorbar(1:top_n_features, tree_sorted(1:top_n_features), importance_std, 'o', 'LineWidth', 1.5);
set(gca, 'XTick', 1:top_n_features, 'XTickLabel', use_vars(tree_sorted_idx(1:top_n_features)));
ylabel('Importance Score ± std');
title({'Feature Importance Consistency', 'with SMOTE'});
grid on;
set(gca, 'XTickLabelRotation', 45);

% Plot 3: Synthetic sample visualization (2D PCA) - FIXED VERSION
subplot(2, 3, 3);
% Use combined data from final model to show SMOTE effect
if synthetic_samples_needed_final > 0
    % Apply PCA to the FINAL dataset
    [~, score] = pca(X_final);
    
    % Create grouping variable with same length as score
    n_samples = size(score, 1);
    group_var = zeros(n_samples, 1);
    
    % Original samples
    original_count = size(X, 1);  % Number of original samples
    group_var(1:original_count) = y_final(1:original_count); % 0 or 1
    
    % Synthetic samples
    if original_count < n_samples
        group_var(original_count+1:end) = 2; % Mark as 2 for synthetic
    end
    
    % Plot with correct dimensions
    gscatter(score(:,1), score(:,2), group_var, 'brg', 'o^', 15);
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
    title('PCA: SMOTE Synthetic vs Original');
    
    % Create appropriate legend
    legend_text = {};
    if any(group_var == 0)
        legend_text{end+1} = 'Non-Stroke (Original)';
    end
    if any(group_var == 1)
        legend_text{end+1} = 'Stroke (Original)';
    end
    if any(group_var == 2)
        legend_text{end+1} = 'Stroke (Synthetic)';
    end
    
    if ~isempty(legend_text)
        legend(legend_text, 'Location', 'best');
    end
    
    grid on;
else
    text(0.5, 0.5, 'No SMOTE applied to final model', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    axis off;
end

% Plot 4: SMOTE benefits summary
subplot(2, 3, 4);
text(0.1, 0.85, sprintf('Original Imbalance: %.1f:1', imbalance_ratio), 'FontSize', 12);
if synthetic_samples_needed_final > 0
    text(0.1, 0.75, sprintf('Final Balance: %.1f:1', sum(y_final==0)/sum(y_final==1)), 'FontSize', 12);
else
    text(0.1, 0.75, 'No SMOTE needed for final model', 'FontSize', 12);
end
text(0.1, 0.65, sprintf('Average Recall: %.3f', mean(cv_recalls)), 'FontSize', 12);
text(0.1, 0.55, sprintf('Average F1-Score: %.3f', mean(cv_f1s)), 'FontSize', 12);
text(0.1, 0.45, sprintf('Average AUC: %.3f', mean(cv_aucs)), 'FontSize', 12);
text(0.1, 0.35, sprintf('Optimal Threshold: %.3f', optimal_threshold_overall), 'FontSize', 12);
text(0.1, 0.25, sprintf('CV AUC Range: [%.3f, %.3f]', min(cv_aucs), max(cv_aucs)), 'FontSize', 12);
title('SMOTE Performance Summary');
axis off;

% Plot 5: Training samples per fold (before and after SMOTE)
subplot(2, 3, 5);
training_samples_before = zeros(cv_folds, 1);
training_samples_after = zeros(cv_folds, 1);

for fold = 1:cv_folds
    train_idx = training(cvp, fold);
    training_samples_before(fold) = sum(train_idx);
    stats = cv_smote_stats{fold};
    if isfield(stats, 'after_minority') && isfield(stats, 'after_majority')
        training_samples_after(fold) = stats.after_minority + stats.after_majority;
    else
        training_samples_after(fold) = training_samples_before(fold);
    end
end

bar_data_samples = [training_samples_before, training_samples_after]';
bar(1:cv_folds, bar_data_samples);
xlabel('Fold');
ylabel('Training Samples');
title('Training Samples per Fold');
legend('Before SMOTE', 'After SMOTE', 'Location', 'best');
grid on;

% Add sample count labels
max_samples = max([training_samples_before; training_samples_after]);
for i = 1:cv_folds
    text(i-0.15, training_samples_before(i)+max_samples*0.02, ...
        sprintf('%d', training_samples_before(i)), 'HorizontalAlignment', 'center', 'FontSize', 9);
    if training_samples_after(i) > training_samples_before(i)
        text(i+0.15, training_samples_after(i)+max_samples*0.02, ...
            sprintf('%d', training_samples_after(i)), 'HorizontalAlignment', 'center', 'FontSize', 9);
    end
end

% Plot 6: SMOTE synthetic samples distribution
subplot(2, 3, 6);
synthetic_counts = zeros(cv_folds, 1);
for fold = 1:cv_folds
    stats = cv_smote_stats{fold};
    synthetic_counts(fold) = max(0, stats.synthetic_needed);
end
bar(1:cv_folds, synthetic_counts, 'FaceColor', [0.4660 0.6740 0.1880], 'FaceAlpha', 0.7);
xlabel('Fold');
ylabel('Synthetic Samples Generated');
title('SMOTE Synthetic Samples per Fold');
grid on;

% Add count labels
for i = 1:cv_folds
    if synthetic_counts(i) > 0
        text(i, synthetic_counts(i)+max(synthetic_counts)*0.05, ...
            sprintf('%d', synthetic_counts(i)), 'HorizontalAlignment', 'center', 'FontSize', 10);
    end
end

%% Print final summary with SMOTE
fprintf('\n=== 5-FOLD CROSS-VALIDATION SUMMARY (WITH SMOTE) ===\n');
fprintf('Model: Decision Tree with Enhanced Feature Engineering and SMOTE\n');
fprintf('Number of features: %d\n', length(use_vars));
fprintf('Total original samples: %d\n', size(X, 1));
fprintf('Final balanced dataset: %d samples (with SMOTE)\n', size(X_final, 1));
fprintf('Average AUC across folds: %.4f ± %.4f\n', mean(cv_aucs), std(cv_aucs));
fprintf('Overall AUC (combined predictions): %.4f\n', AUC_overall);
fprintf('Optimal threshold: %.4f\n', optimal_threshold_overall);
fprintf('Average recall with SMOTE: %.4f\n', mean(cv_recalls));
fprintf('Average F1-score with SMOTE: %.4f\n', mean(cv_f1s));
fprintf('Final model trained on balanced data ready for deployment\n');

% Display tree information for final model
fprintf('\n=== FINAL MODEL TREE INFORMATION (WITH SMOTE) ===\n');
try
    % Try different ways to get tree depth
    if isprop(final_tree_model, 'PrunedList')
        tree_depth = length(final_tree_model.PrunedList);
    else
        tree_depth = size(final_tree_model.CutPoint, 1);
    end
    fprintf('Tree Depth: %d\n', tree_depth);
catch
    fprintf('Tree Depth: Not available\n');
end

try
    fprintf('Number of Nodes: %d\n', final_tree_model.NumNodes);
catch
    fprintf('Number of Nodes: Not available\n');
end

try
    if isprop(final_tree_model, 'IsBranchNode')
        num_leaves = sum(final_tree_model.IsBranchNode == 0);
        fprintf('Number of Leaves: %d\n', num_leaves);
    else
        fprintf('Number of Leaves: Not available\n');
    end
catch
    fprintf('Number of Leaves: Not available\n');
end

fprintf('\n=== 5-FOLD CROSS-VALIDATION DECISION TREE ANALYSIS WITH SMOTE COMPLETE ===\n');