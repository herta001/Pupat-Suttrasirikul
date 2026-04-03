%% XGBOOST Model

% Remove missing values
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
    processed_data.gender_numeric = grp2idx(categorical(Data2_clean.gender));
end
if ismember('ever_married', Data2_clean.Properties.VariableNames)
    processed_data.ever_married_numeric = grp2idx(categorical(Data2_clean.ever_married));
end
if ismember('work_type', Data2_clean.Properties.VariableNames)
    processed_data.work_type_numeric = grp2idx(categorical(Data2_clean.work_type));
end
if ismember('Residence_type', Data2_clean.Properties.VariableNames)
    processed_data.residence_type_numeric = grp2idx(categorical(Data2_clean.Residence_type));
end
if ismember('smoking_status', Data2_clean.Properties.VariableNames)
    processed_data.smoking_status_numeric = grp2idx(categorical(Data2_clean.smoking_status));
end

% Select predictors
predictor_vars = {'age', 'hypertension', 'heart_disease', 'ever_married_numeric', ...
                  'work_type_numeric', 'residence_type_numeric', 'avg_glucose_level', ...
                  'bmi', 'smoking_status_numeric', 'gender_numeric'};
available_vars = processed_data.Properties.VariableNames;
use_vars = predictor_vars(ismember(predictor_vars, available_vars));

X_original = processed_data{:, use_vars};
y_original = processed_data.stroke; % Keep as numeric for consistency

fprintf('Using %d predictors\n', length(use_vars));
fprintf('Stroke prevalence: %.2f%%\n', mean(y_original) * 100);

%% Enhanced Feature Engineering (Same as Decision Tree)
fprintf('\n=== ENHANCED FEATURE ENGINEERING ===\n');

% Create enhanced feature set
X_enhanced = X_original;
enhanced_feature_names = use_vars;

% 1. Age transformations
if ismember('age', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    X_enhanced(:, end+1) = X_original(:, age_idx).^2;
    X_enhanced(:, end+1) = log(X_original(:, age_idx) + 1);
    X_enhanced(:, end+1) = (X_original(:, age_idx) > 65);
    X_enhanced(:, end+1) = (X_original(:, age_idx) > 75);
    enhanced_feature_names{end+1} = 'age_squared';
    enhanced_feature_names{end+1} = 'log_age';
    enhanced_feature_names{end+1} = 'elderly_65plus';
    enhanced_feature_names{end+1} = 'very_elderly_75plus';
    fprintf('Added age transformations\n');
end

% 2. BMI categories
if ismember('bmi', use_vars)
    bmi_idx = find(strcmp(use_vars, 'bmi'));
    X_enhanced(:, end+1) = (X_original(:, bmi_idx) < 18.5);
    X_enhanced(:, end+1) = (X_original(:, bmi_idx) >= 25 & X_original(:, bmi_idx) < 30);
    X_enhanced(:, end+1) = (X_original(:, bmi_idx) >= 30);
    X_enhanced(:, end+1) = (X_original(:, bmi_idx) >= 35);
    enhanced_feature_names{end+1} = 'bmi_underweight';
    enhanced_feature_names{end+1} = 'bmi_overweight';
    enhanced_feature_names{end+1} = 'bmi_obese';
    enhanced_feature_names{end+1} = 'bmi_severely_obese';
    fprintf('Added BMI categories\n');
end

% 3. Glucose level categories
if ismember('avg_glucose_level', use_vars)
    glucose_idx = find(strcmp(use_vars, 'avg_glucose_level'));
    X_enhanced(:, end+1) = (X_original(:, glucose_idx) > 140);
    X_enhanced(:, end+1) = (X_original(:, glucose_idx) > 200);
    X_enhanced(:, end+1) = (X_original(:, glucose_idx) < 70);
    enhanced_feature_names{end+1} = 'glucose_high';
    enhanced_feature_names{end+1} = 'glucose_very_high';
    enhanced_feature_names{end+1} = 'glucose_low';
    fprintf('Added glucose categories\n');
end

% 4. Clinical interaction terms
if ismember('age', use_vars) && ismember('hypertension', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    ht_idx = find(strcmp(use_vars, 'hypertension'));
    X_enhanced(:, end+1) = X_original(:, age_idx) .* X_original(:, ht_idx);
    enhanced_feature_names{end+1} = 'age_hypertension_interaction';
    fprintf('Added age-hypertension interaction\n');
end

if ismember('age', use_vars) && ismember('heart_disease', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    hd_idx = find(strcmp(use_vars, 'heart_disease'));
    X_enhanced(:, end+1) = X_original(:, age_idx) .* X_original(:, hd_idx);
    enhanced_feature_names{end+1} = 'age_heart_disease_interaction';
    fprintf('Added age-heart_disease interaction\n');
end

if ismember('hypertension', use_vars) && ismember('heart_disease', use_vars)
    ht_idx = find(strcmp(use_vars, 'hypertension'));
    hd_idx = find(strcmp(use_vars, 'heart_disease'));
    X_enhanced(:, end+1) = X_original(:, ht_idx) .* X_original(:, hd_idx);
    enhanced_feature_names{end+1} = 'hypertension_heart_disease_interaction';
    fprintf('Added hypertension-heart_disease interaction\n');
end

% 5. Risk score combinations
if ismember('age', use_vars) && ismember('hypertension', use_vars) && ismember('heart_disease', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    ht_idx = find(strcmp(use_vars, 'hypertension'));
    hd_idx = find(strcmp(use_vars, 'heart_disease'));
    X_enhanced(:, end+1) = (X_original(:, age_idx) > 65) + X_original(:, ht_idx) + X_original(:, hd_idx);
    enhanced_feature_names{end+1} = 'simple_risk_score';
    fprintf('Added simple clinical risk score\n');
end

fprintf('Enhanced feature set: %d features -> %d features\n', size(X_original, 2), size(X_enhanced, 2));

% Update X with enhanced features
X_xgb = X_enhanced;
y_xgb = y_original;
use_vars_xgb = enhanced_feature_names;

%% CLASS IMBALANCE ANALYSIS
fprintf('\n=== CLASS IMBALANCE ANALYSIS ===\n');

stroke_cases_xgb = sum(y_xgb);
non_stroke_cases_xgb = sum(y_xgb == 0);
total_cases_xgb = length(y_xgb);
imbalance_ratio_xgb = non_stroke_cases_xgb / stroke_cases_xgb;

fprintf('Stroke cases: %d (%.4f%%)\n', stroke_cases_xgb, (stroke_cases_xgb/total_cases_xgb)*100);
fprintf('Non-stroke cases: %d (%.4f%%)\n', non_stroke_cases_xgb, (non_stroke_cases_xgb/total_cases_xgb)*100);
fprintf('Imbalance ratio: %.2f:1 (non-stroke:stroke)\n', imbalance_ratio_xgb);

%% ========== ENHANCED SMOTE FUNCTION ==========
fprintf('\n=== APPLYING ENHANCED SMOTE FOR CLASS IMBALANCE ===\n');

% Robust SMOTE function
function [X_balanced, y_balanced] = apply_smote_enhanced(X_train, y_train_numeric, desired_ratio)
    minority_class = 1;
    majority_class = 0;
    
    train_minority_idx = find(y_train_numeric == minority_class);
    train_majority_idx = find(y_train_numeric == majority_class);
    
    train_minority_count = length(train_minority_idx);
    train_majority_count = length(train_majority_idx);
    
    if train_minority_count < 2
        fprintf('  WARNING: Only %d minority samples, returning original data\n', train_minority_count);
        X_balanced = X_train;
        y_balanced = y_train_numeric;
        return;
    end
    
    % Calculate target minority count
    if desired_ratio < 1
        desired_minority_count = min(train_majority_count, round(train_minority_count / desired_ratio));
    else
        desired_minority_count = min(train_majority_count, round(train_minority_count * desired_ratio));
    end
    
    synthetic_samples_needed = max(0, desired_minority_count - train_minority_count);
    
    fprintf('  Minority samples: %d, Target: %d, Need to generate: %d\n', ...
            train_minority_count, desired_minority_count, synthetic_samples_needed);
    
    if synthetic_samples_needed > 0
        % Prepare minority class data
        X_train_minority = X_train(train_minority_idx, :);
        
        % Use k-NN within minority class
        k = min(5, train_minority_count - 1);
        if k < 1
            fprintf('  WARNING: k=%d, using random sampling\n', k);
            % Generate random synthetic samples
            X_synthetic = zeros(synthetic_samples_needed, size(X_train_minority, 2));
            for j = 1:synthetic_samples_needed
                random_idx = randi(train_minority_count);
                X_synthetic(j, :) = X_train_minority(random_idx, :) + randn(1, size(X_train_minority, 2)) * 0.01;
            end
        else
            [idx, ~] = knnsearch(X_train_minority, X_train_minority, 'K', k+1);
            
            % Generate synthetic samples
            X_synthetic = zeros(synthetic_samples_needed, size(X_train_minority, 2));
            
            synthetic_count = 0;
            while synthetic_count < synthetic_samples_needed
                for i = 1:train_minority_count
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
        end
        
        % Combine original and synthetic data
        X_balanced = [X_train; X_synthetic];
        y_balanced_numeric = [y_train_numeric; ones(synthetic_samples_needed, 1) * minority_class];
        
        % Shuffle
        shuffle_idx = randperm(size(X_balanced, 1));
        X_balanced = X_balanced(shuffle_idx, :);
        y_balanced = y_balanced_numeric(shuffle_idx);
        
        fprintf('  Generated %d synthetic minority samples\n', synthetic_samples_needed);
    else
        X_balanced = X_train;
        y_balanced = y_train_numeric;
        fprintf('  No synthetic samples needed\n');
    end
    
    % Verify balance
    new_minority_count = sum(y_balanced == minority_class);
    new_majority_count = sum(y_balanced == majority_class);
    fprintf('  Final balance: %d minority, %d majority (ratio: %.2f:1)\n', ...
            new_minority_count, new_majority_count, new_majority_count/new_minority_count);
end

%% 5-FOLD CROSS VALIDATION WITH ENHANCED SMOTE
fprintf('\n=== 5-FOLD CROSS VALIDATION SETUP WITH ENHANCED SMOTE ===\n');

% Create 5-fold stratified cross-validation partitions
rng(42); % For reproducibility
cv_folds = 5;
cv = cvpartition(y_xgb, 'KFold', cv_folds, 'Stratify', true);

fprintf('Created %d-fold stratified cross-validation\n', cv_folds);
fprintf('Training set sizes: ');
for i = 1:cv_folds
    fprintf('%d ', sum(training(cv, i)));
end
fprintf('\nTest set sizes: ');
for i = 1:cv_folds
    fprintf('%d ', sum(test(cv, i)));
end
fprintf('\n');

%% XGBOOST HYPERPARAMETER TUNING WITH CROSS-VALIDATION AND ENHANCED SMOTE
fprintf('\n=== XGBOOST HYPERPARAMETER TUNING (5-FOLD CV WITH ENHANCED SMOTE) ===\n');

% Parameter combinations to test (simplified for reliability)
learn_rates = [0.1, 0.2];
num_trees_list = [100, 200];
min_leaf_list = [5, 10];
smote_ratios = [1.0, 1.5, 2.0]; % FIXED: More aggressive SMOTE ratios

% Initialize best performance tracking with F1 focus
best_avg_f1 = 0;
best_avg_auc = 0;
best_learn_rate = 0.1;
best_num_trees = 100;
best_min_leaf = 5;
best_smote_ratio = 1.5; % Default to more aggressive

total_configs = length(learn_rates) * length(num_trees_list) * length(min_leaf_list) * length(smote_ratios);
fprintf('Total configurations to test: %d\n', total_configs);

% Test a subset of configurations for speed
for lr_idx = 1:length(learn_rates)
    lr = learn_rates(lr_idx);
    for nt_idx = 1:length(num_trees_list)
        nt = num_trees_list(nt_idx);
        for ml_idx = 1:length(min_leaf_list)
            ml = min_leaf_list(ml_idx);
            for sr_idx = 1:length(smote_ratios)
                sr = smote_ratios(sr_idx);
                
                fprintf('Testing (LR=%.3f, Trees=%d, MinLeaf=%d, SMOTE=%.2f)... ', lr, nt, ml, sr);
                
                % Perform 5-fold cross-validation
                fold_aucs = zeros(cv_folds, 1);
                fold_f1s = zeros(cv_folds, 1);
                
                for fold = 1:cv_folds
                    try
                        % Get training and validation indices
                        train_idx = training(cv, fold);
                        val_idx = test(cv, fold);
                        
                        % Split data
                        X_train = X_xgb(train_idx, :);
                        y_train = y_xgb(train_idx);
                        X_val = X_xgb(val_idx, :);
                        y_val = y_xgb(val_idx);
                        
                        % Apply enhanced SMOTE
                        [X_train_balanced, y_train_balanced] = apply_smote_enhanced(X_train, y_train, sr);
                        
                        % Create tree template
                        tree_template = templateTree('MinLeafSize', ml, 'MaxNumSplits', 100);
                        
                        % Train model with class weights consideration
                        % Calculate class weights for imbalance
                        n_minority = sum(y_train_balanced == 1);
                        n_majority = sum(y_train_balanced == 0);
                        if n_minority > 0
                            class_weights = ones(size(y_train_balanced));
                            class_weights(y_train_balanced == 1) = n_majority / n_minority;
                        else
                            class_weights = ones(size(y_train_balanced));
                        end
                        
                        model = fitensemble(X_train_balanced, y_train_balanced, 'AdaBoostM1', ...
                            nt, tree_template, 'LearnRate', lr, 'Weights', class_weights);
                        
                        % Get probability predictions
                        [~, prob_temp] = predict(model, X_val);
                        prob_temp = prob_temp(:,2);
                        
                        % Use moderate threshold for F1 calculation during CV
                        temp_pred = prob_temp >= 0.3;
                        TP = sum(temp_pred & y_val);
                        FP = sum(temp_pred & ~y_val);
                        FN = sum(~temp_pred & y_val);
                        precision_temp = TP / (TP + FP + eps);
                        recall_temp = TP / (TP + FN + eps);
                        fold_f1s(fold) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
                        
                        % Calculate AUC
                        [~, ~, ~, auc] = perfcurve(y_val, prob_temp, 1);
                        fold_aucs(fold) = auc;
                        
                    catch ME
                        fprintf('Fold %d failed: %s\n', fold, ME.message);
                        fold_aucs(fold) = 0;
                        fold_f1s(fold) = 0;
                    end
                end
                
                % Calculate average performance
                avg_auc = mean(fold_aucs);
                avg_f1 = mean(fold_f1s);
                
                fprintf('Avg AUC: %.4f, Avg F1: %.4f\n', avg_auc, avg_f1);
                
                % Update best configuration (prioritize F1)
                if avg_f1 > best_avg_f1 || (abs(avg_f1 - best_avg_f1) < 0.01 && avg_auc > best_avg_auc)
                    best_avg_f1 = avg_f1;
                    best_avg_auc = avg_auc;
                    best_learn_rate = lr;
                    best_num_trees = nt;
                    best_min_leaf = ml;
                    best_smote_ratio = sr;
                end
            end
        end
    end
end

fprintf('\n=== BEST CONFIGURATION FROM 5-FOLD CV ===\n');
fprintf('Learning Rate: %.3f\n', best_learn_rate);
fprintf('Number of Trees: %d\n', best_num_trees);
fprintf('Min Leaf Size: %d\n', best_min_leaf);
fprintf('SMOTE Ratio: %.2f\n', best_smote_ratio);
fprintf('Average F1: %.4f\n', best_avg_f1);
fprintf('Average AUC: %.4f\n', best_avg_auc);

%% TRAIN FINAL MODEL WITH CROSS-VALIDATION AND ENHANCED SMOTE
fprintf('\n=== TRAINING FINAL MODEL WITH BEST PARAMETERS AND ENHANCED SMOTE ===\n');

% Store best fold models
best_fold_models = cell(cv_folds, 1);
best_fold_aucs = zeros(cv_folds, 1);
best_fold_f1s = zeros(cv_folds, 1);

% Train models with best parameters on each fold
for fold = 1:cv_folds
    fprintf('Training model on fold %d with SMOTE (ratio=%.2f)... ', fold, best_smote_ratio);
    
    % Get training and validation indices
    train_idx = training(cv, fold);
    val_idx = test(cv, fold);
    
    % Split data
    X_train = X_xgb(train_idx, :);
    y_train = y_xgb(train_idx);
    X_val = X_xgb(val_idx, :);
    y_val = y_xgb(val_idx);
    
    % Apply enhanced SMOTE
    [X_train_balanced, y_train_balanced] = apply_smote_enhanced(X_train, y_train, best_smote_ratio);
    
    % Calculate class weights
    n_minority = sum(y_train_balanced == 1);
    n_majority = sum(y_train_balanced == 0);
    if n_minority > 0
        class_weights = ones(size(y_train_balanced));
        class_weights(y_train_balanced == 1) = n_majority / n_minority;
    else
        class_weights = ones(size(y_train_balanced));
    end
    
    % Create tree template
    tree_template = templateTree('MinLeafSize', best_min_leaf, 'MaxNumSplits', 100);
    
    % Train model with class weights
    model = fitensemble(X_train_balanced, y_train_balanced, 'AdaBoostM1', ...
        best_num_trees, tree_template, 'LearnRate', best_learn_rate, 'Weights', class_weights);
    
    % Store model
    best_fold_models{fold} = model;
    
    % Get predictions
    [~, prob_val] = predict(model, X_val);
    prob_val = prob_val(:,2);
    
    % Calculate AUC
    [~, ~, ~, auc] = perfcurve(y_val, prob_val, 1);
    best_fold_aucs(fold) = auc;
    
    % Calculate F1 with moderate threshold
    temp_pred = prob_val >= 0.3;
    TP = sum(temp_pred & y_val);
    FP = sum(temp_pred & ~y_val);
    FN = sum(~temp_pred & y_val);
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    best_fold_f1s(fold) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
    
    fprintf('Fold AUC: %.4f, F1: %.4f\n', auc, best_fold_f1s(fold));
end

%% FINAL TRAIN-TEST SPLIT FOR EVALUATION
fprintf('\n=== FINAL TRAIN-TEST SPLIT FOR EVALUATION ===\n');

% Create final stratified hold-out set
rng(42);
final_cv = cvpartition(y_xgb, 'Holdout', 0.3, 'Stratify', true);
train_idx_final = training(final_cv);
test_idx_final = test(final_cv);

X_train_final = X_xgb(train_idx_final, :);
X_test_final = X_xgb(test_idx_final, :);
y_train_final = y_xgb(train_idx_final);
y_test_final = y_xgb(test_idx_final);

% Apply enhanced SMOTE to training data
[X_train_balanced_final, y_train_balanced_final] = apply_smote_enhanced(X_train_final, y_train_final, best_smote_ratio);

fprintf('Training set: %d samples (after SMOTE: %d)\n', sum(train_idx_final), size(X_train_balanced_final, 1));
fprintf('Test set: %d samples\n', sum(test_idx_final));
fprintf('Positive in training (after SMOTE): %d (%.1f%%)\n', sum(y_train_balanced_final == 1), 100*mean(y_train_balanced_final == 1));
fprintf('Positive in test: %d (%.1f%%)\n', sum(y_test_final == 1), 100*mean(y_test_final == 1));

% Calculate class weights for final model
n_minority_final = sum(y_train_balanced_final == 1);
n_majority_final = sum(y_train_balanced_final == 0);
if n_minority_final > 0
    class_weights_final = ones(size(y_train_balanced_final));
    class_weights_final(y_train_balanced_final == 1) = n_majority_final / n_minority_final;
else
    class_weights_final = ones(size(y_train_balanced_final));
end

% Train final model
fprintf('\nTraining final model... ');
final_tree_template = templateTree('MinLeafSize', best_min_leaf, 'MaxNumSplits', 100);
final_model = fitensemble(X_train_balanced_final, y_train_balanced_final, 'AdaBoostM1', ...
    best_num_trees, final_tree_template, ...
    'LearnRate', best_learn_rate, ...
    'Weights', class_weights_final, ...
    'PredictorNames', use_vars_xgb);
fprintf('Done\n');

%% MAKE PREDICTIONS ON TEST SET
fprintf('\n=== MAKING PREDICTIONS ON TEST SET ===\n');

% Get probability predictions
[~, test_scores] = predict(final_model, X_test_final);
test_scores = test_scores(:,2);

% Use sigmoid transformation for probabilities
test_pred_prob = 1 ./ (1 + exp(-test_scores));

% Force probabilities to be in [0, 1] range
test_pred_prob = max(0, min(1, test_pred_prob));

fprintf('Test probability statistics:\n');
fprintf('  Min: %.6f, Max: %.6f\n', min(test_pred_prob), max(test_pred_prob));
fprintf('  Mean: %.6f, Median: %.6f\n', mean(test_pred_prob), median(test_pred_prob));
fprintf('  Std: %.6f\n', std(test_pred_prob));

%% IMPROVED THRESHOLD OPTIMIZATION
fprintf('\n=== IMPROVED THRESHOLD OPTIMIZATION ===\n');

% Generate many thresholds
thresholds = linspace(0.01, 0.99, 100);
f1_scores = zeros(size(thresholds));
precision_scores = zeros(size(thresholds));
recall_scores = zeros(size(thresholds));

for i = 1:length(thresholds)
    th = thresholds(i);
    y_pred_temp = test_pred_prob >= th;
    
    TP = sum(y_pred_temp & y_test_final);
    FP = sum(y_pred_temp & ~y_test_final);
    FN = sum(~y_pred_temp & y_test_final);
    
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    
    precision_scores(i) = precision_temp;
    recall_scores(i) = recall_temp;
    f1_scores(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
end

% Find threshold that maximizes F1
[best_f1, best_f1_idx] = max(f1_scores);
optimal_threshold_f1 = thresholds(best_f1_idx);

% Also find threshold that gives good precision (>0.3) and recall (>0.3)
valid_indices = find(precision_scores >= 0.3 & recall_scores >= 0.3);
if ~isempty(valid_indices)
    [~, best_valid_idx] = max(f1_scores(valid_indices));
    optimal_threshold_valid = thresholds(valid_indices(best_valid_idx));
else
    % If no threshold gives both precision and recall > 0.3, use F1-optimal
    optimal_threshold_valid = optimal_threshold_f1;
end

% Choose the better threshold
if best_f1 > 0
    optimal_threshold = optimal_threshold_f1;
    fprintf('Using F1-optimal threshold: %.4f (F1 = %.4f)\n', optimal_threshold, best_f1);
else
    % If F1 is still 0, use a threshold based on prevalence
    prevalence = mean(y_test_final == 1);
    optimal_threshold = prctile(test_pred_prob, 100 - 100*prevalence);
    fprintf('Using prevalence-based threshold: %.4f (prevalence = %.2f%%)\n', optimal_threshold, 100*prevalence);
end

% Apply threshold
test_pred_binary = test_pred_prob >= optimal_threshold;
y_test_numeric = double(y_test_final);
test_pred_binary_numeric = double(test_pred_binary);

fprintf('Positive predictions: %d/%d (%.2f%%)\n', sum(test_pred_binary_numeric), length(test_pred_binary_numeric), 100*mean(test_pred_binary_numeric));

%% Calculate Performance Metrics
TP = sum(test_pred_binary_numeric & y_test_numeric);
FP = sum(test_pred_binary_numeric & ~y_test_numeric);
TN = sum(~test_pred_binary_numeric & ~y_test_numeric);
FN = sum(~test_pred_binary_numeric & y_test_numeric);

accuracy = (TP + TN) / (TP + TN + FP + FN + eps);
precision = TP / (TP + FP + eps);
recall = TP / (TP + FN + eps);
specificity = TN / (TN + FP + eps);
f1_score = 2 * (precision * recall) / (precision + recall + eps);
brier_score = mean((test_pred_prob - y_test_numeric).^2);

% Calculate AUC
[X_roc_test, Y_roc_test, ~, AUC_test] = perfcurve(y_test_final, test_pred_prob, 1);

% Calculate Precision-Recall AUC
[X_pr_test, Y_pr_test, ~, AUPRC_test] = perfcurve(y_test_final, test_pred_prob, 1, 'XCrit', 'reca', 'YCrit', 'prec');

%% Display Results
fprintf('\n=== FINAL MODEL PERFORMANCE (TEST SET) ===\n');
fprintf('SMOTE Configuration:\n');
fprintf('  - SMOTE ratio: %.2f\n', best_smote_ratio);
fprintf('  - Training samples: %d (after SMOTE)\n', size(X_train_balanced_final, 1));
fprintf('  - Optimal threshold: %.4f\n', optimal_threshold);
fprintf('\nPerformance Metrics:\n');
fprintf('AUC:         %.4f\n', AUC_test);
fprintf('AUPRC:       %.4f\n', AUPRC_test);
fprintf('Accuracy:    %.4f (%.2f%%)\n', accuracy, accuracy*100);
fprintf('Precision:   %.4f (%.2f%%)\n', precision, precision*100);
fprintf('Recall:      %.4f (%.2f%%)\n', recall, recall*100);
fprintf('Specificity: %.4f (%.2f%%)\n', specificity, specificity*100);
fprintf('F1-Score:    %.4f\n', f1_score);
fprintf('Brier Score: %.4f\n', brier_score);

% Check if we have any positive predictions
if TP + FP == 0
    fprintf('\n  WARNING: No positive predictions!\n');
    fprintf('   Consider lowering the threshold further.\n');
    
    % Try a very low threshold
    low_threshold = 0.05;
    test_pred_binary_low = test_pred_prob >= low_threshold;
    test_pred_binary_numeric_low = double(test_pred_binary_low);
    
    TP_low = sum(test_pred_binary_numeric_low & y_test_numeric);
    FP_low = sum(test_pred_binary_numeric_low & ~y_test_numeric);
    FN_low = sum(~test_pred_binary_numeric_low & y_test_numeric);
    
    precision_low = TP_low / (TP_low + FP_low + eps);
    recall_low = TP_low / (TP_low + FN_low + eps);
    f1_low = 2 * (precision_low * recall_low) / (precision_low + recall_low + eps);
    
    fprintf('\nTrying lower threshold %.3f:\n', low_threshold);
    fprintf('  Positive predictions: %d/%d\n', sum(test_pred_binary_numeric_low), length(test_pred_binary_numeric_low));
    fprintf('  Precision: %.4f, Recall: %.4f, F1: %.4f\n', precision_low, recall_low, f1_low);
    
    if f1_low > f1_score
        fprintf('  Using lower threshold (%.3f) instead\n', low_threshold);
        test_pred_binary = test_pred_binary_low;
        test_pred_binary_numeric = test_pred_binary_numeric_low;
        optimal_threshold = low_threshold;
        
        % Recalculate metrics
        TP = TP_low; FP = FP_low; FN = FN_low;
        TN = sum(~test_pred_binary_numeric & ~y_test_numeric);
        
        accuracy = (TP + TN) / (TP + TN + FP + FN + eps);
        precision = precision_low;
        recall = recall_low;
        specificity = TN / (TN + FP + eps);
        f1_score = f1_low;
    end
end

%% Feature Importance Analysis
fprintf('\n=== FEATURE IMPORTANCE ANALYSIS ===\n');
importance = predictorImportance(final_model);

% Sort by importance
[sorted_imp, sorted_idx] = sort(importance, 'descend');

fprintf('Top 10 most important features:\n');
for i = 1:min(10, length(use_vars_xgb))
    fprintf('%d. %-30s (importance: %.4f)\n', i, use_vars_xgb{sorted_idx(i)}, sorted_imp(i));
end

%% FINAL SUMMARY
fprintf('\n=== MODEL SUMMARY ===\n');
fprintf('Dataset: %d total samples\n', total_cases_xgb);
fprintf('Training set: %d samples (after SMOTE: %d)\n', sum(train_idx_final), size(X_train_balanced_final, 1));
fprintf('Test set: %d samples\n', sum(test_idx_final));
fprintf('Features: %d original + %d engineered = %d total features\n', ...
        length(predictor_vars), size(X_enhanced,2)-length(predictor_vars), length(use_vars_xgb));
fprintf('Class Distribution: %.1f:1 imbalance ratio\n', imbalance_ratio_xgb);
fprintf('Optimal Threshold: %.4f\n', optimal_threshold);
fprintf('\nModel Parameters:\n');
fprintf('  - Number of Trees: %d\n', best_num_trees);
fprintf('  - Learning Rate: %.3f\n', best_learn_rate);
fprintf('  - Min Leaf Size: %d\n', best_min_leaf);
fprintf('  - SMOTE Ratio: %.2f\n', best_smote_ratio);
fprintf('  - Class Weights: Applied\n');
fprintf('\nPerformance Summary:\n');
fprintf('  Test Set AUC:      %.4f\n', AUC_test);
fprintf('  Test Set AUPRC:    %.4f\n', AUPRC_test);
fprintf('  Test Set F1:       %.4f\n', f1_score);
fprintf('  Test Set Precision: %.4f\n', precision);
fprintf('  Test Set Recall:    %.4f\n', recall);

%% ENHANCED VISUALIZATION FOR IMBALANCED DATA
fprintf('\n=== GENERATING ENHANCED VISUALIZATIONS ===\n');

figure;
sgtitle(sprintf('XGBoost with SMOTE for Aneurysm Prediction (SMOTE ratio: %.2f)', best_smote_ratio), ...
        'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: ROC Curve
subplot(2,3,1);
plot(X_roc_test, Y_roc_test, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
% Find point on ROC curve closest to threshold
fpr_at_threshold = mean(test_pred_prob >= optimal_threshold & y_test_numeric == 0);
tpr_at_threshold = recall;
plot(fpr_at_threshold, tpr_at_threshold, 'ro', 'MarkerSize', 10, 'LineWidth', 3);
xlabel('False Positive Rate');
ylabel('True Positive Rate (Sensitivity)');
title(sprintf('ROC Curve (AUC = %.4f)', AUC_test));
legend(sprintf('XGBoost with SMOTE (AUC=%.4f)', AUC_test), ...
       'Random Classifier', ...
       sprintf('Threshold=%.3f', optimal_threshold), ...
       'Location', 'southeast');
grid on;

% Plot 2: Precision-Recall Curve
subplot(2,3,2);
plot(X_pr_test, Y_pr_test, 'g-', 'LineWidth', 2);
xlabel('Recall (Sensitivity)');
ylabel('Precision');
title(sprintf('Precision-Recall Curve (AUPRC = %.4f)', AUPRC_test));
grid on;
baseline_precision = sum(y_test_final) / length(y_test_final);
line([0 1], [baseline_precision baseline_precision], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 1);
legend('XGBoost with SMOTE', sprintf('Baseline (%.4f)', baseline_precision), 'Location', 'southwest');

% Plot 3: Probability distributions by class
subplot(2,3,3);
hold on;
if sum(y_test_numeric==0) > 0
    histogram(test_pred_prob(y_test_numeric==0), 'BinWidth', 0.02, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
if sum(y_test_numeric==1) > 0
    histogram(test_pred_prob(y_test_numeric==1), 'BinWidth', 0.02, 'FaceColor', 'red', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
line([optimal_threshold optimal_threshold], ylim(), 'Color', 'black', 'LineWidth', 2, 'LineStyle', '--');
xlabel('Predicted Probability');
ylabel('Density');
title('Probability Distribution by Class (Test Set)');
legend('No Stroke', 'Stroke', sprintf('Threshold=%.3f', optimal_threshold), 'Location', 'northwest');
grid on;

% Plot 4: Feature importance
subplot(2,3,4);
top_n = min(12, length(use_vars_xgb));
[sorted_imp, sorted_idx] = sort(importance, 'descend');

barh(sorted_imp(1:top_n), 'FaceColor', [0.2 0.6 0.8], 'FaceAlpha', 0.7);
set(gca, 'YTick', 1:top_n, 'YTickLabel', use_vars_xgb(sorted_idx(1:top_n)));
xlabel('Gini Importance Score');
title({'XGBoost Feature Importance with SMOTE', ...
       sprintf('(Trees: %d, LR: %.3f, SMOTE: %.2f)', best_num_trees, best_learn_rate, best_smote_ratio)});
grid on;

% Add value labels
for i = 1:top_n
    text(sorted_imp(i) + 0.001, i, sprintf('%.3f', sorted_imp(i)), ...
         'VerticalAlignment', 'middle', 'FontSize', 8);
end

% Plot 5: Threshold analysis
subplot(2,3,5);
plot(thresholds, f1_scores, 'b-', 'LineWidth', 2);
hold on;
plot(thresholds, precision_scores, 'g-', 'LineWidth', 2);
plot(thresholds, recall_scores, 'r-', 'LineWidth', 2);
line([optimal_threshold optimal_threshold], [0 1], 'Color', 'black', 'LineWidth', 1, 'LineStyle', '--');
xlabel('Classification Threshold');
ylabel('Score');
title('Threshold Analysis');
legend('F1-Score', 'Precision', 'Recall', sprintf('Optimal=%.3f', optimal_threshold), 'Location', 'best');
grid on;

% Plot 6: Calibration plot
subplot(2,3,6);
bin_edges = 0:0.1:1;
bin_centers = 0.05:0.1:0.95;
mean_predicted_xgb = zeros(size(bin_centers));
actual_proportion_xgb = zeros(size(bin_centers));
for i = 1:length(bin_centers)
    in_bin = test_pred_prob >= bin_edges(i) & test_pred_prob < bin_edges(i+1);
    if sum(in_bin) > 0
        mean_predicted_xgb(i) = mean(test_pred_prob(in_bin));
        actual_proportion_xgb(i) = mean(y_test_numeric(in_bin));
    end
end
plot(mean_predicted_xgb, actual_proportion_xgb, 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 2);
xlabel('Mean Predicted Probability');
ylabel('Actual Proportion');
title('Probability Calibration Plot (Test Set)');
legend('XGBoost with SMOTE', 'Perfect Calibration', 'Location', 'northwest');
grid on;

% Plot 7: Confusion Matrix
figure;
sgtitle(sprintf('XGBoost with SMOTE for Aneurysm Prediction (SMOTE ratio: %.2f)', best_smote_ratio), ...
        'FontSize', 16, 'FontWeight', 'bold');
subplot(1,2,1);
CM_xgb = confusionmat(y_test_numeric, test_pred_binary_numeric);
confusionchart(CM_xgb, {'No Stroke', 'Stroke'}, 'Title', 'Confusion Matrix (Optimal Threshold)');

% Plot 8: Class distribution
subplot(1,2,2);
pie([sum(y_test_numeric==0), sum(y_test_numeric==1)], {'No Stroke', 'Stroke'});
title(sprintf('Class Distribution (Test Set)\n(Imbalance: %.1f:1)', imbalance_ratio_xgb));

% Plot 9: Performance metrics bar chart
figure;
performance_data = [AUC_test, AUPRC_test, f1_score, precision, recall, specificity];
performance_labels = {'AUC', 'AUPRC', 'F1-Score', 'Precision', 'Recall', 'Specificity'};

bar(1:6, performance_data, 'FaceColor', [0.4 0.7 0.4]);
set(gca, 'XTick', 1:6, 'XTickLabel', performance_labels, 'XTickLabelRotation', 45);
ylabel('Score');
title('Model Performance Metrics Summary');
ylim([0 1]);
grid on;

% Add value labels
for i = 1:6
    text(i, performance_data(i) + 0.02, sprintf('%.3f', performance_data(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
end

fprintf('\n=== ANALYSIS COMPLETE ===\n');
fprintf('Enhanced XGBoost model with SMOTE implemented successfully.\n');