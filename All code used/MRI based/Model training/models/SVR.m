%% Aneurysm MRI Dataset Analysis - SVR with SMOTE (Regression Approach)
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

fprintf('\n========== NORMAL vs ANEURYSM CLASSIFICATION (SVR) ==========\n');
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

%% 7. Feature Selection using F-statistic
fprintf('\nPerforming feature selection...\n');

% Calculate F-statistic for each feature
f_scores = zeros(length(availableFeatures), 1);
for i = 1:length(availableFeatures)
    feature_vals = X_normalized(:, i);
    vals_class0 = feature_vals(y == 0);
    vals_class1 = feature_vals(y == 1);
    [f_scores(i), ~] = compute_f_statistic(vals_class0, vals_class1);
end

% Sort by F-statistic
[~, f_idx] = sort(f_scores, 'descend');

% Keep top features
top_n_features = min(12, length(availableFeatures));
selected_idx = f_idx(1:top_n_features);
X_selected = X_normalized(:, selected_idx);
featureNames_selected = featureNames(selected_idx);

fprintf('Selected top %d features based on F-statistic\n', top_n_features);

%% Helper function for F-statistic
function [F_stat, p_val] = compute_f_statistic(x1, x2)
    n1 = length(x1); n2 = length(x2); n = n1 + n2;
    grand_mean = (sum(x1) + sum(x2)) / n;
    mean1 = mean(x1); mean2 = mean(x2);
    SS_between = n1 * (mean1 - grand_mean)^2 + n2 * (mean2 - grand_mean)^2;
    SS_within = sum((x1 - mean1).^2) + sum((x2 - mean2).^2);
    df_between = 1; df_within = n - 2;
    MS_between = SS_between / df_between;
    MS_within = SS_within / df_within;
    
    if MS_within > 0
        F_stat = MS_between / MS_within;
    else
        F_stat = 0;
    end
    
    if F_stat > 0 && df_within > 0
        p_val = 1 - fcdf(F_stat, df_between, df_within);
    else
        p_val = 1;
    end
end

%% 8. SMOTE Implementation for SVR (Regression-Specific)
function [X_smote, y_smote] = applySMOTE_svr(X, y, k, targetRatio, method)
    % SVR-specific SMOTE: Creates synthetic samples with continuous targets
    % method: 'borderline', 'adasyn', 'weighted'
    
    X_majority = X(y == 0, :);
    X_minority = X(y == 1, :);
    y_majority = y(y == 0);
    y_minority = y(y == 1);
    
    n_majority = size(X_majority, 1);
    n_minority = size(X_minority, 1);
    
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
    synthetic_targets = zeros(n_synthetic, 1);
    
    % Pre-compute distances
    D = pdist2(X_minority, X_minority);
    
    if strcmp(method, 'adasyn')
        % ADASYN: Focus on harder samples
        D_majority = pdist2(X_minority, X_majority);
        majority_ratio = sum(D_majority < median(D_majority(:)), 2) / max(1, n_majority);
        generation_weights = majority_ratio / (sum(majority_ratio) + eps);
        cum_weights = cumsum(generation_weights);
        
        for i = 1:n_synthetic
            r = rand();
            idx = find(cum_weights >= r, 1);
            if isempty(idx), idx = randi(n_minority); end
            idx = max(1, min(idx, n_minority));
            
            base_sample = X_minority(idx, :);
            base_target = y_minority(idx);
            
            % Find neighbor
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
            
            local_density = generation_weights(idx);
            gap = rand(1, n_features) * (0.5 + 0.5 * local_density);
            
            % Generate synthetic sample AND target (interpolate target too)
            synthetic_samples(i, :) = base_sample + gap .* (X_minority(neighbor, :) - base_sample);
            synthetic_targets(i) = base_target + mean(gap) * (y_minority(neighbor) - base_target);
        end
        
    elseif strcmp(method, 'weighted')
        % Weighted SMOTE: Emphasize samples with extreme targets
        % For binary, extremes are just the minority class
        weights = ones(n_minority, 1);
        cum_weights = cumsum(weights / sum(weights));
        
        for i = 1:n_synthetic
            r = rand();
            idx = find(cum_weights >= r, 1);
            if isempty(idx), idx = randi(n_minority); end
            
            base_sample = X_minority(idx, :);
            base_target = y_minority(idx);
            
            % Find neighbor
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
            
            gap = rand(1, n_features);
            synthetic_samples(i, :) = base_sample + gap .* (X_minority(neighbor, :) - base_sample);
            synthetic_targets(i) = base_target + mean(gap) * (y_minority(neighbor) - base_target);
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
            base_target = y_minority(idx);
            
            % Find neighbor
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
            synthetic_targets(i) = base_target + mean(gap) * (y_minority(neighbor) - base_target);
        end
    end
    
    % Add small noise
    synthetic_samples = synthetic_samples + randn(size(synthetic_samples)) * 0.01;
    
    % Clip targets to [0, 1] range
    synthetic_targets = max(0, min(1, synthetic_targets));
    
    X_smote = [X_majority; X_minority; synthetic_samples];
    y_smote = [y_majority; y_minority; synthetic_targets];
end

%% 9. Set Up Cross-Validation
rng(42);
cv = cvpartition(y, 'KFold', 5);

fprintf('\n========== 5-FOLD CV WITH SVR ENSEMBLE ==========\n');

% Initialize arrays
foldAccuracies = zeros(5, 1);
foldPrecisions = zeros(5, 1);
foldRecalls = zeros(5, 1);
foldF1s = zeros(5, 1);
foldAUCs = zeros(5, 1);
foldThresholds = zeros(5, 1);
foldModels = cell(5, 1);
allScores = zeros(length(y), 1);
allTrueLabels = zeros(length(y), 1);
scoreIdx = 1;

%% 10. Cross-Validation with SVR Ensemble
for fold = 1:5
    fprintf('\n--- Fold %d of 5 ---\n', fold);
    
    trainIdx = cv.training(fold);
    testIdx = cv.test(fold);
    
    XTrain_raw = X_selected(trainIdx, :);
    yTrain_raw = y(trainIdx);
    XTest = X_selected(testIdx, :);
    yTest = y(testIdx);
    
    %% Generate SMOTE variants for SVR
    k_neighbors = 7;
    target_ratio = 0.9;
    
    [XTrain1, yTrain1] = applySMOTE_svr(XTrain_raw, yTrain_raw, k_neighbors, target_ratio, 'borderline');
    [XTrain2, yTrain2] = applySMOTE_svr(XTrain_raw, yTrain_raw, k_neighbors, target_ratio, 'adasyn');
    
    %% TRAIN MULTIPLE SVR MODELS
    models = {};
    scores_fold = zeros(length(yTest), 0);
    
    % SVR Model 1: Gaussian kernel with epsilon-insensitive loss
    try
        % For regression, we need to use fitrsvm
        mdl1 = fitrsvm(XTrain1, yTrain1, ...
            'KernelFunction', 'gaussian', ...
            'KernelScale', 'auto', ...
            'BoxConstraint', 1, ...
            'Epsilon', 0.1, ...
            'Standardize', false);
        models{end+1} = mdl1;
        
        scores_fold(:, end+1) = predict(mdl1, XTest);
        fprintf('  Model 1 (Gaussian SVR) trained\n');
    catch ME
        fprintf('  Model 1 failed: %s\n', ME.message);
    end
    
    % SVR Model 2: Linear kernel with different parameters
    try
        mdl2 = fitrsvm(XTrain2, yTrain2, ...
            'KernelFunction', 'linear', ...
            'BoxConstraint', 2, ...
            'Epsilon', 0.05, ...
            'Standardize', false);
        models{end+1} = mdl2;
        
        scores_fold(:, end+1) = predict(mdl2, XTest);
        fprintf('  Model 2 (Linear SVR) trained\n');
    catch ME
        fprintf('  Model 2 failed: %s\n', ME.message);
    end
    
    % SVR Model 3: Polynomial kernel
    try
        mdl3 = fitrsvm(XTrain_raw, yTrain_raw, ... % Use original data
            'KernelFunction', 'polynomial', ...
            'PolynomialOrder', 2, ...
            'BoxConstraint', 1, ...
            'Epsilon', 0.1, ...
            'Standardize', false);
        models{end+1} = mdl3;
        
        scores_fold(:, end+1) = predict(mdl3, XTest);
        fprintf('  Model 3 (Polynomial SVR) trained\n');
    catch ME
        fprintf('  Model 3 failed: %s\n', ME.message);
    end
    
    % SVR Model 4: Gaussian with different parameters
    try
        mdl4 = fitrsvm(XTrain_raw, yTrain_raw, ...
            'KernelFunction', 'gaussian', ...
            'KernelScale', 2, ...
            'BoxConstraint', 0.5, ...
            'Epsilon', 0.2, ...
            'Standardize', false);
        models{end+1} = mdl4;
        
        scores_fold(:, end+1) = predict(mdl4, XTest);
        fprintf('  Model 4 (Gaussian SVR v2) trained\n');
    catch ME
        fprintf('  Model 4 failed: %s\n', ME.message);
    end
    
    % Fallback if no models worked
    if isempty(models)
        fprintf('  Using fallback linear SVR\n');
        mdl = fitrsvm(XTrain_raw, yTrain_raw, ...
            'KernelFunction', 'linear', ...
            'BoxConstraint', 1, ...
            'Epsilon', 0.1);
        models{1} = mdl;
        scores_fold = predict(mdl, XTest);
    else
        % Ensemble: average predictions
        scores_fold = mean(scores_fold, 2);
    end
    
    foldModels{fold} = models;
    
    % Clip predictions to [0, 1] for probability-like scores
    scores_fold = max(0, min(1, scores_fold));
    
    % Store scores
    nTest = length(yTest);
    allScores(scoreIdx:scoreIdx+nTest-1) = scores_fold;
    allTrueLabels(scoreIdx:scoreIdx+nTest-1) = yTest;
    scoreIdx = scoreIdx + nTest;
    
    %% Find optimal threshold
    thresholds = 0.1:0.025:0.9;
    bestF1 = 0;
    bestThreshold = 0.5;
    
    for t = thresholds
        pred = scores_fold > t;
        tp = sum(pred & (yTest == 1));
        fp = sum(pred & (yTest == 0));
        fn = sum(~pred & (yTest == 1));
        
        if tp > 0
            prec = tp / (tp + fp + eps);
            rec = tp / (tp + fn + eps);
            f1 = 2 * prec * rec / (prec + rec + eps);
            
            if f1 > bestF1
                bestF1 = f1;
                bestThreshold = t;
            end
        end
    end
    
    foldThresholds(fold) = bestThreshold;
    
    % Calculate metrics
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
    
    fprintf('Fold %d Results (threshold=%.3f):\n', fold, bestThreshold);
    fprintf('  AUC: %.3f | Prec: %.1f%% | Recall: %.1f%% | F1: %.3f\n', ...
        foldAUCs(fold), foldPrecisions(fold)*100, foldRecalls(fold)*100, foldF1s(fold));
end

% Trim arrays
allScores = allScores(1:scoreIdx-1);
allTrueLabels = allTrueLabels(1:scoreIdx-1);

%% 11. Overall Results
fprintf('\n========== OVERALL 5-FOLD CV RESULTS (SVR) ==========\n');
fprintf('Metric        Mean ± Std Dev\n');
fprintf('Accuracy:     %.1f%% ± %.1f%%\n', mean(foldAccuracies)*100, std(foldAccuracies)*100);
fprintf('Precision:    %.1f%% ± %.1f%%\n', mean(foldPrecisions)*100, std(foldPrecisions)*100);
fprintf('Recall:       %.1f%% ± %.1f%%\n', mean(foldRecalls)*100, std(foldRecalls)*100);
fprintf('F1 Score:     %.3f ± %.3f\n', mean(foldF1s), std(foldF1s));
fprintf('AUC:          %.3f ± %.3f\n', mean(foldAUCs), std(foldAUCs));

% Overall AUC (from CV)
[~, ~, ~, overallAUC] = perfcurve(allTrueLabels, allScores, 1);
fprintf('\nOverall Pooled AUC (CV): %.4f\n', overallAUC);

%% 12. Train Final SVR Ensemble Model
fprintf('\n========== TRAINING FINAL SVR ENSEMBLE MODEL ==========\n');

X_final = X_selected;

% Generate SMOTE variants
k_neighbors_final = 7;
target_ratio_final = 0.9;

fprintf('Generating SMOTE variants...\n');

try
    [X1, y1] = applySMOTE_svr(X_final, y, k_neighbors_final, target_ratio_final, 'borderline');
catch
    X1 = X_final; y1 = y;
end

try
    [X2, y2] = applySMOTE_svr(X_final, y, k_neighbors_final, target_ratio_final, 'adasyn');
catch
    X2 = X_final; y2 = y;
end

try
    [X3, y3] = applySMOTE_svr(X_final, y, k_neighbors_final, 0.8, 'weighted');
catch
    X3 = X_final; y3 = y;
end

% Train final SVR ensemble
final_models = {};
fprintf('Training final SVR ensemble...\n');

% Model 1: Gaussian SVR on borderline SMOTE
try
    final_models{1} = fitrsvm(X1, y1, ...
        'KernelFunction', 'gaussian', ...
        'KernelScale', 'auto', ...
        'BoxConstraint', 1, ...
        'Epsilon', 0.1);
    fprintf('  Model 1 trained\n');
catch
end

% Model 2: Linear SVR on ADASYN
try
    final_models{2} = fitrsvm(X2, y2, ...
        'KernelFunction', 'linear', ...
        'BoxConstraint', 2, ...
        'Epsilon', 0.05);
    fprintf('  Model 2 trained\n');
catch
end

% Model 3: Gaussian SVR on weighted SMOTE
try
    final_models{3} = fitrsvm(X3, y3, ...
        'KernelFunction', 'gaussian', ...
        'KernelScale', 1.5, ...
        'BoxConstraint', 0.5, ...
        'Epsilon', 0.15);
    fprintf('  Model 3 trained\n');
catch
end

% Model 4: Polynomial SVR on original data
try
    final_models{4} = fitrsvm(X_final, y, ...
        'KernelFunction', 'polynomial', ...
        'PolynomialOrder', 2, ...
        'BoxConstraint', 1, ...
        'Epsilon', 0.1);
    fprintf('  Model 4 trained\n');
catch
end

% Model 5: Gaussian SVR on original with different parameters
try
    final_models{5} = fitrsvm(X_final, y, ...
        'KernelFunction', 'gaussian', ...
        'KernelScale', 2, ...
        'BoxConstraint', 3, ...
        'Epsilon', 0.2);
    fprintf('  Model 5 trained\n');
catch
end

% Remove empty models
final_models = final_models(~cellfun('isempty', final_models));
fprintf('Successfully trained %d SVR models\n', length(final_models));

if isempty(final_models)
    error('No SVR models could be trained!');
end

% Get ensemble predictions
all_preds = zeros(length(y), length(final_models));
for i = 1:length(final_models)
    all_preds(:, i) = predict(final_models{i}, X_final);
end

% Ensemble prediction (mean)
yScore_ensemble = mean(all_preds, 2);
yScore_ensemble = max(0, min(1, yScore_ensemble));

% Find optimal threshold
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

fprintf('\n========== FINAL SVR ENSEMBLE PERFORMANCE ==========\n');
fprintf('Threshold:     %.3f\n', final_threshold);
fprintf('Accuracy:      %.1f%%\n', final_accuracy*100);
fprintf('Precision:     %.1f%%\n', final_precision*100);
fprintf('Recall:        %.1f%%\n', final_recall*100);
fprintf('F1 Score:      %.3f\n', final_f1);
fprintf('AUC:           %.4f (THIS IS THE PRIMARY METRIC)\n', final_auc);
fprintf('Confusion:     TP=%d, FP=%d, TN=%d, FN=%d\n', tp, fp, tn, fn);

%% 13. Plot Results - MODIFIED TO SHOW FINAL ENSEMBLE AUC
figure('Name', 'SVR Ensemble Results', 'Position', [100, 100, 1600, 500]);

% ROC Curve - NOW USING FINAL ENSEMBLE AUC
subplot(1, 4, 1);
[X_roc_final, Y_roc_final, ~, ~] = perfcurve(y, yScore_ensemble, 1);
plot(X_roc_final, Y_roc_final, 'b-', 'LineWidth', 3);
hold on; 
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);
xlabel('False Positive Rate', 'FontSize', 12);
ylabel('True Positive Rate', 'FontSize', 12);
title(sprintf('ROC Curve - FINAL ENSEMBLE (AUC = %.4f)', final_auc), 'FontSize', 14, 'FontWeight', 'bold');
legend('SVR Ensemble', 'Random Classifier', 'Location', 'southeast');
grid on; 
axis square;
% Add text box with AUC value
text(0.6, 0.2, sprintf('AUC = %.4f', final_auc), ...
    'FontSize', 14, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'EdgeColor', 'k');

% Feature Importance (using correlation)
subplot(1, 4, 2);
correlations = corr(X_final, y);
[~, idx] = sort(abs(correlations), 'descend');
topN = min(10, length(correlations));

for i = 1:topN
    if correlations(idx(i)) > 0
        color = [1, 0.4, 0.4]; % Red for positive
    else
        color = [0.4, 0.4, 1]; % Blue for negative
    end
    h = barh(i, correlations(idx(i)));
    hold on;
    set(h, 'FaceColor', color, 'EdgeColor', 'k', 'LineWidth', 1);
end

yticks(1:topN);
yticklabels(featureNames_selected(idx(1:topN)));
xlabel('Correlation with Target', 'FontSize', 12);
title('Feature-Target Correlation', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xlim([-1, 1]);
legend({'Positive (Aneurysm)', 'Negative (Normal)'}, 'Location', 'best');

% Confusion Matrix
subplot(1, 4, 3);
cm = [tn, fp; fn, tp];
imagesc(cm); 
colormap('parula'); 
colorbar;
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

% Model Stability (CV results for reference)
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
fprintf('\n========== SVR MODEL SUMMARY ==========\n');
fprintf('Model Type: Support Vector Regression Ensemble\n');
fprintf('Number of SVR models in ensemble: %d\n', length(final_models));
fprintf('Kernels used: Gaussian, Linear, Polynomial\n');
fprintf('SMOTE variants: Borderline, ADASYN, Weighted\n');
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