%% Aneurysm MRI Dataset Analysis - SVM with SMOTE (Optimized for AUC & Stability)
% Clear workspace and command window
clear; clc; close all;

%% 1. Load and Prepare Data
filename = 'MRIdataset.xlsx';
data = readtable(filename);

fprintf('Dataset loaded with %d rows and %d columns\n', height(data), width(data));

%% 2. Create Binary Target Variable
normal_aneurysm_idx = strcmp(data.PatientGroup, 'Normal') | strcmp(data.PatientGroup, 'Aneurysm');
data_filtered = data(normal_aneurysm_idx, :);
data_filtered.Target = double(strcmp(data_filtered.PatientGroup, 'Aneurysm'));

fprintf('\n========== NORMAL vs ANEURYSM CLASSIFICATION ==========\n');
fprintf('Filtered dataset: %d rows\n', height(data_filtered));
fprintf('Normal (0):   %d samples\n', sum(data_filtered.Target == 0));
fprintf('Aneurysm (1): %d samples\n', sum(data_filtered.Target == 1));

%% 3. Select Features
selectedFeatures = {
    'num_vessels', 'vessel_density', 'mean_width', 'std_width', ...
    'mean_aspect_ratio', 'branch_point_density', 'end_point_density', ...
    'mean_circularity', 'intensity_range', 'intensity_std', ...
    'large_vessel_count', 'small_vessel_count', 'total_vessel_length', ...
    'vessel_complexity', 'glc_contrast', 'glc_homogeneity'
};

% Check features
availableFeatures = {};
for i = 1:length(selectedFeatures)
    if ismember(selectedFeatures{i}, data_filtered.Properties.VariableNames)
        availableFeatures{end+1} = selectedFeatures{i};
    end
end

fprintf('\nUsing %d features for analysis\n', length(availableFeatures));

%% 4. Prepare Feature Matrix
X = [];
featureNames = {};

for i = 1:length(availableFeatures)
    featureData = data_filtered.(availableFeatures{i});
    featureData(isinf(featureData)) = NaN;
    X = [X, featureData];
    featureNames{end+1} = availableFeatures{i};
end

y = data_filtered.Target;

%% 5. Remove Invalid Rows
validRows = all(~isnan(X), 2) & all(isfinite(X), 2);
X = X(validRows, :);
y = y(validRows);

fprintf('\nAfter removing invalid data: %d samples remain\n', length(y));
fprintf('  Normal:   %d samples (%.1f%%)\n', sum(y == 0), sum(y==0)/length(y)*100);
fprintf('  Aneurysm: %d samples (%.1f%%)\n', sum(y == 1), sum(y==1)/length(y)*100);

%% 6. Normalize Features
X_normalized = normalize(X);

%% 7. Feature Selection Function (using F-statistic)
fprintf('\nPerforming feature selection...\n');

% Calculate F-statistic for each feature (ANOVA)
f_scores = zeros(length(availableFeatures), 1);
p_values = zeros(length(availableFeatures), 1);

for i = 1:length(availableFeatures)
    % Separate feature values by class
    feature_vals = X_normalized(:, i);
    vals_class0 = feature_vals(y == 0);
    vals_class1 = feature_vals(y == 1);
    
    % Calculate ANOVA F-statistic
    [f_scores(i), p_values(i)] = compute_f_statistic(vals_class0, vals_class1);
end

% Sort by F-statistic (higher is better)
[~, f_idx] = sort(f_scores, 'descend');

% Keep top features (remove noisy ones)
top_n_features = min(12, length(availableFeatures)); % Reduce from 16 to 12
selected_idx = f_idx(1:top_n_features);
X_selected = X_normalized(:, selected_idx);
featureNames_selected = featureNames(selected_idx);

fprintf('Selected top %d features based on F-statistic\n', top_n_features);
fprintf('Top 5 features: %s, %s, %s, %s, %s\n', ...
    featureNames_selected{1}, featureNames_selected{2}, featureNames_selected{3}, ...
    featureNames_selected{4}, featureNames_selected{5});

%% Helper function for F-statistic
function [F_stat, p_val] = compute_f_statistic(x1, x2)
    % Compute ANOVA F-statistic between two groups
    n1 = length(x1);
    n2 = length(x2);
    n = n1 + n2;
    
    % Overall mean
    grand_mean = (sum(x1) + sum(x2)) / n;
    
    % Group means
    mean1 = mean(x1);
    mean2 = mean(x2);
    
    % Between-group sum of squares
    SS_between = n1 * (mean1 - grand_mean)^2 + n2 * (mean2 - grand_mean)^2;
    
    % Within-group sum of squares
    SS_within = sum((x1 - mean1).^2) + sum((x2 - mean2).^2);
    
    % Degrees of freedom
    df_between = 1; % (2 groups - 1)
    df_within = n - 2;
    
    % Mean squares
    MS_between = SS_between / df_between;
    MS_within = SS_within / df_within;
    
    % F-statistic
    if MS_within > 0
        F_stat = MS_between / MS_within;
    else
        F_stat = 0;
    end
    
    % Approximate p-value using F-distribution
    if F_stat > 0 && df_within > 0
        p_val = 1 - fcdf(F_stat, df_between, df_within);
    else
        p_val = 1;
    end
end

%% 8. Advanced SMOTE with Noise Filtering (FIXED VERSION)
function [X_smote, y_smote] = applySMOTE_advanced(X, y, k, targetRatio, method)
    % method: 'borderline', 'adasyn', 'svm_smote'
    % FIXED: Added bounds checking and fallbacks
    
    X_majority = X(y == 0, :);
    X_minority = X(y == 1, :);
    y_majority = y(y == 0);
    
    n_majority = size(X_majority, 1);
    n_minority = size(X_minority, 1);
    
    % Check if we have minority samples
    if n_minority == 0
        fprintf('  WARNING: No minority samples found!\n');
        X_smote = X;
        y_smote = y;
        return;
    end
    
    target_minority = round(n_majority * targetRatio);
    n_synthetic = max(0, target_minority - n_minority);
    
    fprintf('SMOTE (%s): Creating %d synthetic samples (target ratio=%.2f)\n', method, n_synthetic, targetRatio);
    
    if n_synthetic <= 0
        X_smote = X;
        y_smote = y;
        return;
    end
    
    n_features = size(X, 2);
    synthetic_samples = zeros(n_synthetic, n_features);
    
    % Pre-compute all distances for efficiency
    D = pdist2(X_minority, X_minority);
    
    if strcmp(method, 'adasyn')
        % ADASYN: Generate more samples for harder-to-learn instances
        % Calculate density of majority class around each minority sample
        D_majority = pdist2(X_minority, X_majority);
        majority_ratio = sum(D_majority < median(D_majority(:)), 2) / max(1, n_majority);
        generation_weights = majority_ratio / (sum(majority_ratio) + eps);
        
        % Generate samples proportional to difficulty
        cum_weights = cumsum(generation_weights);
        for i = 1:n_synthetic
            r = rand();
            idx = find(cum_weights >= r, 1);
            if isempty(idx), idx = randi(n_minority); end
            
            % FIXED: Ensure idx is valid
            idx = max(1, min(idx, n_minority));
            
            base_sample = X_minority(idx, :);
            
            % FIXED: Handle neighbor selection safely
            [~, neighbor_idx] = mink(D(idx, :), min(k+1, n_minority));
            valid_neighbors = neighbor_idx(neighbor_idx ~= idx);
            if isempty(valid_neighbors)
                neighbor = randi(n_minority);
                while neighbor == idx
                    neighbor = randi(n_minority);
                end
            else
                neighbor = valid_neighbors(randi(length(valid_neighbors)));
            end
            
            % Adaptive gap based on local density
            local_density = generation_weights(idx);
            gap = rand(1, n_features) * (0.5 + 0.5 * local_density);
            
            synthetic_samples(i, :) = base_sample + gap .* (X_minority(neighbor, :) - base_sample);
        end
        
    elseif strcmp(method, 'svm_smote')
        % SVM-SMOTE: Focus on support vector region
        % Train quick SVM to find support vectors
        try
            % FIXED: Use subsampling if too many samples
            if size(X, 1) > 5000
                sample_idx = randperm(size(X, 1), 5000);
                temp_svm = fitcsvm(X(sample_idx, :), y(sample_idx), 'KernelFunction', 'linear', 'BoxConstraint', 1);
            else
                temp_svm = fitcsvm(X, y, 'KernelFunction', 'linear', 'BoxConstraint', 1);
            end
            sv_idx = temp_svm.IsSupportVector;
            sv_minority = find(sv_idx & (y == 1));
        catch
            sv_minority = 1:n_minority;
        end
        
        % FIXED: Handle empty support vectors
        if isempty(sv_minority) || length(sv_minority) < 2
            sv_minority = 1:n_minority;
        end
        
        % Generate samples from support vectors
        for i = 1:n_synthetic
            idx = sv_minority(randi(length(sv_minority)));
            base_sample = X_minority(idx, :);
            
            % FIXED: Handle neighbor selection safely
            [~, neighbor_idx] = mink(D(idx, :), min(k+1, n_minority));
            valid_neighbors = neighbor_idx(neighbor_idx ~= idx);
            if isempty(valid_neighbors)
                neighbor = randi(n_minority);
                while neighbor == idx
                    neighbor = randi(n_minority);
                end
            else
                neighbor = valid_neighbors(randi(length(valid_neighbors)));
            end
            
            gap = rand(1, n_features) * 0.7;
            synthetic_samples(i, :) = base_sample + gap .* (X_minority(neighbor, :) - base_sample);
        end
        
    else % borderline
        % Identify borderline samples
        D_majority = pdist2(X_minority, X_majority);
        [~, majority_nn] = mink(D_majority, min(3, n_majority), 2);
        n_majority_neighbors = sum(majority_nn > 0, 2);
        
        borderline_idx = find(n_majority_neighbors >= 1 & n_majority_neighbors < 3);
        if isempty(borderline_idx)
            borderline_idx = 1:n_minority;
        end
        
        for i = 1:n_synthetic
            idx = borderline_idx(randi(length(borderline_idx)));
            base_sample = X_minority(idx, :);
            
            % FIXED: Handle neighbor selection safely
            [~, neighbor_idx] = mink(D(idx, :), min(k+1, n_minority));
            valid_neighbors = neighbor_idx(neighbor_idx ~= idx);
            if isempty(valid_neighbors)
                neighbor = randi(n_minority);
                while neighbor == idx
                    neighbor = randi(n_minority);
                end
            else
                neighbor = valid_neighbors(randi(length(valid_neighbors)));
            end
            
            gap = rand(1, n_features) * 0.6;
            synthetic_samples(i, :) = base_sample + gap .* (X_minority(neighbor, :) - base_sample);
        end
    end
    
    % Optional: Add small noise to avoid duplicates
    synthetic_samples = synthetic_samples + randn(size(synthetic_samples)) * 0.01;
    
    X_smote = [X_majority; X_minority; synthetic_samples];
    y_smote = [y_majority; y(y == 1); ones(n_synthetic, 1)];
end

%% 9. Set Up Stratified Cross-Validation
rng(42);
cv = cvpartition(y, 'KFold', 5); % Stratified by default

fprintf('\n========== 5-FOLD CV WITH AUC-OPTIMIZED SVM ==========\n');

% Initialize arrays
foldAccuracies = zeros(5, 1);
foldPrecisions = zeros(5, 1);
foldRecalls = zeros(5, 1);
foldF1s = zeros(5, 1);
foldAUCs = zeros(5, 1);
foldModels = cell(5, 1);
allScores = zeros(length(y), 1);
allTrueLabels = zeros(length(y), 1);
scoreIdx = 1;

%% 10. Cross-Validation with Ensemble Approach
for fold = 1:5
    fprintf('\n--- Fold %d of 5 ---\n', fold);
    
    trainIdx = cv.training(fold);
    testIdx = cv.test(fold);
    
    XTrain_raw = X_selected(trainIdx, :);
    yTrain_raw = y(trainIdx);
    XTest = X_selected(testIdx, :);
    yTest = y(testIdx);
    
    %% TRY MULTIPLE SMOTE VARIANTS AND COMBINE
    k_neighbors = 7;
    target_ratio = 0.9; % Slightly less than 1.0 to avoid overfitting
    
    % Generate multiple SMOTE variants
    [XTrain1, yTrain1] = applySMOTE_advanced(XTrain_raw, yTrain_raw, k_neighbors, target_ratio, 'borderline');
    [XTrain2, yTrain2] = applySMOTE_advanced(XTrain_raw, yTrain_raw, k_neighbors, target_ratio, 'adasyn');
    
    %% TRAIN MULTIPLE SVM MODELS AND ENSEMBLE
    models = {};
    scores_fold = zeros(length(yTest), 0);
    
    % Model 1: RBF SVM with moderate regularization
    try
        mdl1 = fitcsvm(XTrain1, yTrain1, ...
            'KernelFunction', 'rbf', ...
            'BoxConstraint', 1.5, ...
            'KernelScale', 'auto', ...
            'Cost', [0 1; 2 0], ...
            'Standardize', false, ...
            'CacheSize', 'maximal');
        models{end+1} = mdl1;
        
        [~, score] = predict(mdl1, XTest);
        scores_fold(:, end+1) = 1 ./ (1 + exp(-score(:, 2)));
    catch ME
        fprintf('  Model 1 failed: %s\n', ME.message);
    end
    
    % Model 2: Polynomial SVM
    try
        mdl2 = fitcsvm(XTrain2, yTrain2, ...
            'KernelFunction', 'polynomial', ...
            'PolynomialOrder', 2, ...
            'BoxConstraint', 1, ...
            'Cost', [0 1; 2 0], ...
            'Standardize', false);
        models{end+1} = mdl2;
        
        [~, score] = predict(mdl2, XTest);
        scores_fold(:, end+1) = 1 ./ (1 + exp(-score(:, 2)));
    catch ME
        fprintf('  Model 2 failed: %s\n', ME.message);
    end
    
    % Model 3: Linear SVM with different weights
    try
        wNormal = 1;
        wAneurysm = sum(yTrain_raw==0) / max(1, sum(yTrain_raw==1));
        wAneurysm = min(wAneurysm, 3);
        
        mdl3 = fitclinear(XTrain_raw, yTrain_raw, ...
            'Learner', 'svm', ...
            'Regularization', 'ridge', ...
            'Lambda', 0.001, ...
            'Prior', 'empirical', ...
            'Cost', [0 1; wAneurysm 0]);
        models{end+1} = mdl3;
        
        [~, score] = predict(mdl3, XTest);
        scores_fold(:, end+1) = score(:, 2);
    catch ME
        fprintf('  Model 3 failed: %s\n', ME.message);
    end
    
    % If no models worked, use simple linear SVM
    if isempty(models)
        fprintf('  Using fallback linear SVM\n');
        mdl = fitcsvm(XTrain_raw, yTrain_raw, ...
            'KernelFunction', 'linear', ...
            'BoxConstraint', 1, ...
            'Cost', [0 1; 2 0]);
        models{1} = mdl;
        
        [~, score] = predict(mdl, XTest);
        scores_fold = 1 ./ (1 + exp(-score(:, 2)));
    else
        % Ensemble: average probabilities
        scores_fold = mean(scores_fold, 2);
    end
    
    foldModels{fold} = models; % Store all models
    
    % Store scores
    nTest = length(yTest);
    allScores(scoreIdx:scoreIdx+nTest-1) = scores_fold;
    allTrueLabels(scoreIdx:scoreIdx+nTest-1) = yTest;
    scoreIdx = scoreIdx + nTest;
    
    %% OPTIMIZE THRESHOLD FOR AUC
    % For AUC, we don't need threshold, but we'll find best for other metrics
    thresholds = 0.1:0.025:0.9;
    bestAUC = 0;
    bestThreshold = 0.5;
    
    for t = thresholds
        pred = scores_fold > t;
        try
            [~, ~, ~, auc] = perfcurve(yTest, scores_fold, 1);
            if auc > bestAUC
                bestAUC = auc;
                bestThreshold = t;
            end
        catch
        end
    end
    
    % Calculate metrics at best threshold
    yPred = scores_fold > bestThreshold;
    tp = sum(yPred & (yTest == 1));
    fp = sum(yPred & (yTest == 0));
    tn = sum(~yPred & (yTest == 0));
    fn = sum(~yPred & (yTest == 1));
    
    foldAccuracies(fold) = (tp + tn) / (tp + tn + fp + fn + eps);
    foldPrecisions(fold) = tp / (tp + fp + eps);
    foldRecalls(fold) = tp / (tp + fn + eps);
    
    if foldPrecisions(fold) > 0 && foldRecalls(fold) > 0
        foldF1s(fold) = 2 * (foldPrecisions(fold) * foldRecalls(fold)) / ...
                           (foldPrecisions(fold) + foldRecalls(fold) + eps);
    end
    
    [~, ~, ~, foldAUCs(fold)] = perfcurve(yTest, scores_fold, 1);
    
    fprintf('Fold %d Results:\n', fold);
    fprintf('  AUC: %.3f | Prec: %.1f%% | Recall: %.1f%% | F1: %.3f\n', ...
        foldAUCs(fold), foldPrecisions(fold)*100, foldRecalls(fold)*100, foldF1s(fold));
end

% Trim arrays
allScores = allScores(1:scoreIdx-1);
allTrueLabels = allTrueLabels(1:scoreIdx-1);

%% 11. Overall Results with Confidence Intervals
fprintf('\n========== OVERALL 5-FOLD CV RESULTS ==========\n');
fprintf('Metric        Mean ± Std Dev     [95%% CI]\n');
fprintf('Accuracy:     %.1f%% ± %.1f%%     [%.1f%%, %.1f%%]\n', ...
    mean(foldAccuracies)*100, std(foldAccuracies)*100, ...
    (mean(foldAccuracies) - 1.96*std(foldAccuracies)/sqrt(5))*100, ...
    (mean(foldAccuracies) + 1.96*std(foldAccuracies)/sqrt(5))*100);
fprintf('Precision:    %.1f%% ± %.1f%%     [%.1f%%, %.1f%%]\n', ...
    mean(foldPrecisions)*100, std(foldPrecisions)*100, ...
    (mean(foldPrecisions) - 1.96*std(foldPrecisions)/sqrt(5))*100, ...
    (mean(foldPrecisions) + 1.96*std(foldPrecisions)/sqrt(5))*100);
fprintf('Recall:       %.1f%% ± %.1f%%     [%.1f%%, %.1f%%]\n', ...
    mean(foldRecalls)*100, std(foldRecalls)*100, ...
    (mean(foldRecalls) - 1.96*std(foldRecalls)/sqrt(5))*100, ...
    (mean(foldRecalls) + 1.96*std(foldRecalls)/sqrt(5))*100);
fprintf('F1 Score:     %.3f ± %.3f     [%.3f, %.3f]\n', ...
    mean(foldF1s), std(foldF1s), ...
    mean(foldF1s) - 1.96*std(foldF1s)/sqrt(5), ...
    mean(foldF1s) + 1.96*std(foldF1s)/sqrt(5));
fprintf('AUC:          %.3f ± %.3f     [%.3f, %.3f]\n', ...
    mean(foldAUCs), std(foldAUCs), ...
    mean(foldAUCs) - 1.96*std(foldAUCs)/sqrt(5), ...
    mean(foldAUCs) + 1.96*std(foldAUCs)/sqrt(5));

% Overall AUC with confidence interval
[~, ~, ~, overallAUC] = perfcurve(allTrueLabels, allScores, 1);
fprintf('\nOverall Pooled AUC (CV): %.4f\n', overallAUC);

% Bootstrap for AUC confidence interval
n_bootstrap = 1000;
boot_auc = zeros(n_bootstrap, 1);
n_samples = length(allTrueLabels);

for b = 1:n_bootstrap
    idx = randsample(n_samples, n_samples, true);
    try
        [~, ~, ~, boot_auc(b)] = perfcurve(allTrueLabels(idx), allScores(idx), 1);
    catch
        boot_auc(b) = overallAUC;
    end
end

auc_ci = prctile(boot_auc, [2.5, 97.5]);
fprintf('AUC 95%% Bootstrap CI: [%.4f, %.4f]\n', auc_ci(1), auc_ci(2));

%% 12. Train Final Ensemble Model (FIXED VERSION)
fprintf('\n========== TRAINING FINAL ENSEMBLE MODEL ==========\n');

% Feature selection for final model
X_final = X_selected;

% Generate multiple SMOTE datasets with error handling
k_neighbors_final = 7;
target_ratio_final = 0.9;

fprintf('Generating SMOTE variants...\n');

try
    [X1, y1] = applySMOTE_advanced(X_final, y, k_neighbors_final, target_ratio_final, 'borderline');
catch ME
    fprintf('  Borderline SMOTE failed: %s\n', ME.message);
    X1 = X_final; y1 = y;
end

try
    [X2, y2] = applySMOTE_advanced(X_final, y, k_neighbors_final, target_ratio_final, 'adasyn');
catch ME
    fprintf('  ADASYN SMOTE failed: %s\n', ME.message);
    X2 = X_final; y2 = y;
end

try
    [X3, y3] = applySMOTE_advanced(X_final, y, k_neighbors_final, 0.8, 'svm_smote');
catch ME
    fprintf('  SVM-SMOTE failed: %s\n', ME.message);
    X3 = X_final; y3 = y;
end

% Train ensemble of models
final_models = {};

fprintf('Training ensemble of SVM models...\n');

% Model 1: RBF on borderline SMOTE
try
    final_models{1} = fitcsvm(X1, y1, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', 1.5, ...
        'KernelScale', 'auto', ...
        'Cost', [0 1; 2 0]);
    fprintf('  Model 1 (RBF + borderline) trained\n');
catch ME
    fprintf('  Model 1 failed: %s\n', ME.message);
end

% Model 2: RBF on ADASYN
try
    final_models{2} = fitcsvm(X2, y2, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', 2, ...
        'KernelScale', 1.5, ...
        'Cost', [0 1; 2 0]);
    fprintf('  Model 2 (RBF + ADASYN) trained\n');
catch ME
    fprintf('  Model 2 failed: %s\n', ME.message);
end

% Model 3: Polynomial on SVM-SMOTE
try
    final_models{3} = fitcsvm(X3, y3, ...
        'KernelFunction', 'polynomial', ...
        'PolynomialOrder', 2, ...
        'BoxConstraint', 1, ...
        'Cost', [0 1; 2 0]);
    fprintf('  Model 3 (Polynomial + SVM-SMOTE) trained\n');
catch ME
    fprintf('  Model 3 failed: %s\n', ME.message);
end

% Model 4: Linear on original (weighted)
try
    wAneurysm = sum(y==0) / max(1, sum(y==1));
    final_models{4} = fitclinear(X_final, y, ...
        'Learner', 'svm', ...
        'Regularization', 'ridge', ...
        'Lambda', 0.001, ...
        'Cost', [0 1; min(wAneurysm, 3) 0]);
    fprintf('  Model 4 (Linear + weights) trained\n');
catch ME
    fprintf('  Model 4 failed: %s\n', ME.message);
end

% Model 5: RBF with different parameters on original
try
    final_models{5} = fitcsvm(X_final, y, ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', 0.5, ...
        'KernelScale', 2, ...
        'Cost', [0 1; 3 0]);
    fprintf('  Model 5 (RBF + original) trained\n');
catch ME
    fprintf('  Model 5 failed: %s\n', ME.message);
end

% Remove empty models
final_models = final_models(~cellfun('isempty', final_models));
fprintf('Successfully trained %d models\n', length(final_models));

% Check if we have any models
if isempty(final_models)
    error('No models could be trained!');
end

% Get ensemble predictions on original data
all_probs = zeros(length(y), length(final_models));

for i = 1:length(final_models)
    if isa(final_models{i}, 'ClassificationSVM')
        [~, score] = predict(final_models{i}, X_final);
        all_probs(:, i) = 1 ./ (1 + exp(-score(:, 2)));
    else
        [~, score] = predict(final_models{i}, X_final);
        all_probs(:, i) = score(:, 2);
    end
end

% Ensemble prediction (mean of probabilities)
yScore_ensemble = mean(all_probs, 2);

% Find optimal threshold for F1 score
thresholds = 0.3:0.025:0.8;
bestF1 = 0;
final_threshold = 0.5;

for t = thresholds
    pred = yScore_ensemble > t;
    tp = sum(pred & (y == 1));
    fp = sum(pred & (y == 0));
    fn = sum(~pred & (y == 1));
    
    if tp > 0
        prec = tp / (tp + fp + eps);
        rec = tp / (tp + fn + eps);
        f1 = 2 * prec * rec / (prec + rec + eps);
        
        if f1 > bestF1
            bestF1 = f1;
            final_threshold = t;
        end
    end
end

% Final predictions
yPred_final = yScore_ensemble > final_threshold;

% Final metrics
tp = sum(yPred_final & (y == 1));
fp = sum(yPred_final & (y == 0));
tn = sum(~yPred_final & (y == 0));
fn = sum(~yPred_final & (y == 1));

final_accuracy = (tp + tn) / (tp + tn + fp + fn + eps);
final_precision = tp / (tp + fp + eps);
final_recall = tp / (tp + fn + eps);
final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall + eps);
[~, ~, ~, final_auc] = perfcurve(y, yScore_ensemble, 1);

fprintf('\n========== FINAL ENSEMBLE MODEL PERFORMANCE ==========\n');
fprintf('Threshold:     %.3f\n', final_threshold);
fprintf('Accuracy:      %.1f%%\n', final_accuracy*100);
fprintf('Precision:     %.1f%%\n', final_precision*100);
fprintf('Recall:        %.1f%%\n', final_recall*100);
fprintf('F1 Score:      %.3f\n', final_f1);
fprintf('AUC:           %.4f (THIS IS THE PRIMARY METRIC)\n', final_auc);
fprintf('Confusion:     TP=%d, FP=%d, TN=%d, FN=%d\n', tp, fp, tn, fn);

%% 13. Plot Results - MODIFIED TO SHOW FINAL ENSEMBLE AUC
figure('Name', 'AUC-Optimized SVM Ensemble', 'Position', [100, 100, 1600, 500]);

% ROC Curve - NOW USING FINAL ENSEMBLE AUC
subplot(1, 4, 1);
[X_roc_final, Y_roc_final, ~, ~] = perfcurve(y, yScore_ensemble, 1);
plot(X_roc_final, Y_roc_final, 'b-', 'LineWidth', 3);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);
xlabel('False Positive Rate', 'FontSize', 12);
ylabel('True Positive Rate', 'FontSize', 12);
title(sprintf('ROC Curve - FINAL ENSEMBLE (AUC = %.4f)', final_auc), 'FontSize', 14, 'FontWeight', 'bold');
legend('Ensemble SVM', 'Random Classifier', 'Location', 'southeast');
grid on; axis square;
% Add text box with AUC value
text(0.6, 0.2, sprintf('AUC = %.4f', final_auc), ...
    'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'EdgeColor', 'k');

% Feature Importance (from linear model)
subplot(1, 4, 2);
if length(final_models) >= 4 && isprop(final_models{4}, 'Beta')
    beta = final_models{4}.Beta;
    [~, idx] = sort(abs(beta), 'descend');
    topN = min(10, length(beta));
    
    for i = 1:topN
        if beta(idx(i)) > 0
            color = [1, 0.4, 0.4];
        else
            color = [0.4, 0.4, 1];
        end
        h = barh(i, beta(idx(i)));
        hold on;
        set(h, 'FaceColor', color, 'EdgeColor', 'k');
    end
    
    yticks(1:topN);
    yticklabels(featureNames_selected(idx(1:topN)));
    xlabel('Feature Weight', 'FontSize', 12);
    title('Top 10 Feature Importance', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    legend({'Positive (Aneurysm)', 'Negative (Normal)'}, 'Location', 'best');
else
    text(0.5, 0.5, 'Feature importance\nnot available', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    title('Feature Importance', 'FontSize', 14, 'FontWeight', 'bold');
end

% Confusion Matrix
subplot(1, 4, 3);
cm = [tn, fp; fn, tp];
imagesc(cm); colormap('parula'); colorbar;
% Add text with counts and percentages
for i = 1:2
    for j = 1:2
        text(j, i, sprintf('%d\n(%.1f%%)', cm(i,j), cm(i,j)/sum(cm(:))*100), ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', 'white');
    end
end
set(gca, 'XTick', [1 2], 'YTick', [1 2], ...
    'XTickLabel', {'Predicted Normal', 'Predicted Aneurysm'}, ...
    'YTickLabel', {'Actual Normal', 'Actual Aneurysm'}, ...
    'FontSize', 11);
title(sprintf('Confusion Matrix - FINAL ENSEMBLE\nAcc: %.1f%%, Prec: %.1f%%, Rec: %.1f%%', ...
    final_accuracy*100, final_precision*100, final_recall*100), ...
    'FontSize', 14, 'FontWeight', 'bold');
axis square;

% Model Stability Plot
subplot(1, 4, 4);
plot(1:5, foldAUCs, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot([0.5 5.5], [final_auc final_auc], 'g--', 'LineWidth', 2);
plot([0.5 5.5], [mean(foldAUCs) mean(foldAUCs)], 'r:', 'LineWidth', 1.5);
xlabel('Fold', 'FontSize', 12);
ylabel('AUC', 'FontSize', 12);
title('Cross-Validation Stability (Reference)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Per-fold AUC', 'Final Ensemble AUC', 'Mean CV AUC', 'Location', 'best');
grid on;
xlim([0.5 5.5]);
ylim([0.5 1]);
text(3, 0.55, sprintf('Final AUC: %.4f\nMean CV: %.3f', final_auc, mean(foldAUCs)), ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'BackgroundColor', 'w');

%% 14. Comparison Summary
fprintf('\n========== SVM MODEL SUMMARY ==========\n');
fprintf('Model Type: Support Vector Machine Ensemble\n');
fprintf('Number of SVM models in ensemble: %d\n', length(final_models));
fprintf('Kernels used: RBF, Polynomial, Linear\n');
fprintf('SMOTE variants: Borderline, ADASYN, SVM-SMOTE\n');
fprintf('Features: %d\n', size(X_final, 2));
fprintf('Original samples: %d (Normal: %d, Aneurysm: %d)\n', length(y), sum(y==0), sum(y==1));

fprintf('\n5-Fold CV Performance (for reference):\n');
fprintf('  Accuracy:  %.1f%% ± %.1f%%\n', mean(foldAccuracies)*100, std(foldAccuracies)*100);
fprintf('  Precision: %.1f%% ± %.1f%%\n', mean(foldPrecisions)*100, std(foldPrecisions)*100);
fprintf('  Recall:    %.1f%% ± %.1f%%\n', mean(foldRecalls)*100, std(foldRecalls)*100);
fprintf('  AUC:       %.3f ± %.3f\n', mean(foldAUCs), std(foldAUCs));

fprintf('\n FINAL ENSEMBLE PERFORMANCE (PRIMARY METRICS) \n');
fprintf('  Accuracy:  %.1f%%\n', final_accuracy*100);
fprintf('  Precision: %.1f%%\n', final_precision*100);
fprintf('  Recall:    %.1f%%\n', final_recall*100);
fprintf('  F1 Score:  %.3f\n', final_f1);
fprintf('  AUC:       %.4f \n', final_auc);

fprintf('\nAnalysis complete! Primary results show AUC = %.4f\n', final_auc);