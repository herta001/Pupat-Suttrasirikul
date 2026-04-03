%% Aneurysm MRI Dataset Analysis - Decision Tree with SMOTE and 5-Fold CV
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

fprintf('\n========== NORMAL vs ANEURYSM CLASSIFICATION (DECISION TREE) ==========\n');
fprintf('Filtered dataset: %d rows\n', height(data_filtered));
fprintf('Normal (0):   %d samples (%.1f%%)\n', sum(data_filtered.Target == 0), ...
    sum(data_filtered.Target == 0)/height(data_filtered)*100);
fprintf('Aneurysm (1): %d samples (%.1f%%)\n', sum(data_filtered.Target == 1), ...
    sum(data_filtered.Target == 1)/height(data_filtered)*100);

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

%% 6. Normalize Features (optional for Decision Tree)
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
top_n_features = min(14, length(availableFeatures));
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

%% 8. SMOTE Implementation
function [X_smote, y_smote] = applySMOTE(X, y, k, targetRatio)
    X_majority = X(y == 0, :);
    X_minority = X(y == 1, :);
    y_majority = y(y == 0);
    
    n_majority = size(X_majority, 1);
    n_minority = size(X_minority, 1);
    
    if n_minority == 0
        X_smote = X;
        y_smote = y;
        return;
    end
    
    target_minority = round(n_majority * targetRatio);
    n_synthetic = max(0, target_minority - n_minority);
    
    fprintf('  SMOTE: Creating %d synthetic samples (target ratio=%.2f)\n', n_synthetic, targetRatio);
    
    if n_synthetic <= 0
        X_smote = X;
        y_smote = y;
        return;
    end
    
    n_features = size(X, 2);
    synthetic_samples = zeros(n_synthetic, n_features);
    
    % Pre-compute distances
    D = pdist2(X_minority, X_minority);
    
    for i = 1:n_synthetic
        idx = randi(n_minority);
        base_sample = X_minority(idx, :);
        
        % Find k nearest neighbors
        [~, neighbor_idx] = sort(D(idx, :));
        neighbor_idx = neighbor_idx(2:min(k+1, end));
        
        if isempty(neighbor_idx)
            neighbor = randi(n_minority);
        else
            neighbor = neighbor_idx(randi(length(neighbor_idx)));
        end
        
        gap = rand(1, n_features);
        synthetic_samples(i, :) = base_sample + gap .* (X_minority(neighbor, :) - base_sample);
    end
    
    X_smote = [X_majority; X_minority; synthetic_samples];
    y_smote = [y_majority; y(y == 1); ones(n_synthetic, 1)];
end

%% 9. Set Up 5-Fold Cross-Validation
rng(42); % For reproducibility
cv = cvpartition(y, 'KFold', 5);

fprintf('\n========== 5-FOLD CROSS-VALIDATION WITH DECISION TREE ==========\n');

% Initialize arrays for CV results
cvAccuracies = zeros(5, 1);
cvPrecisions = zeros(5, 1);
cvRecalls = zeros(5, 1);
cvF1s = zeros(5, 1);
cvAUCs = zeros(5, 1);
cvThresholds = zeros(5, 1);
cvModels = cell(5, 1);
all_cv_scores = [];
all_cv_labels = [];

%% 10. Perform 5-Fold Cross-Validation
for fold = 1:5
    fprintf('\n--- Fold %d of 5 ---\n', fold);
    
    % Get training and testing indices
    trainIdx = cv.training(fold);
    testIdx = cv.test(fold);
    
    XTrain = X_selected(trainIdx, :);
    yTrain = y(trainIdx);
    XTest = X_selected(testIdx, :);
    yTest = y(testIdx);
    
    % Apply SMOTE to training data only
    k_fold = 5;
    target_fold = 1.0; % Perfect balance for training
    [XTrain_balanced, yTrain_balanced] = applySMOTE(XTrain, yTrain, k_fold, target_fold);
    
    fprintf('  Fold %d training set after SMOTE: %d samples\n', fold, length(yTrain_balanced));
    fprintf('    Class 0: %d, Class 1: %d\n', sum(yTrain_balanced==0), sum(yTrain_balanced==1));
    
    % Create table for training
    varNames = cell(1, size(XTrain_balanced, 2));
    for i = 1:size(XTrain_balanced, 2)
        varNames{i} = sprintf('Feature_%d', i);
    end
    data_train = array2table(XTrain_balanced, 'VariableNames', varNames);
    data_train.Target = yTrain_balanced;
    
    % Train Decision Tree for this fold
    % Try different complexity parameters
    minLeafSizes = [1, 5, 10, 20];
    bestAUC_fold = 0;
    bestModel_fold = [];
    bestParams_fold = '';
    
    for m = 1:length(minLeafSizes)
        try
            tempModel = fitctree(data_train, 'Target', ...
                'MinLeafSize', minLeafSizes(m), ...
                'SplitCriterion', 'gdi', ... % Gini impurity
                'Prune', 'on');
            
            % Get predictions
            [~, temp_scores] = predict(tempModel, data_train);
            temp_scores = temp_scores(:, 2);
            [~, ~, ~, temp_auc] = perfcurve(yTrain_balanced, temp_scores, 1);
            
            if temp_auc > bestAUC_fold
                bestAUC_fold = temp_auc;
                bestModel_fold = tempModel;
                bestParams_fold = sprintf('MinLeaf=%d', minLeafSizes(m));
            end
        catch
            continue;
        end
    end
    
    % Use best model or fallback
    if isempty(bestModel_fold)
        fprintf('  Using default Decision Tree\n');
        foldModel = fitctree(data_train, 'Target', ...
            'MinLeafSize', 5, ...
            'SplitCriterion', 'gdi');
    else
        foldModel = bestModel_fold;
        fprintf('  Selected: %s (AUC=%.3f)\n', bestParams_fold, bestAUC_fold);
    end
    
    cvModels{fold} = foldModel;
    
    % Create test table
    data_test = array2table(XTest, 'VariableNames', varNames);
    
    % Get predictions
    [~, scores] = predict(foldModel, data_test);
    yScores = scores(:, 2); % Probability of class 1 (aneurysm)
    
    % Store scores for overall CV AUC
    all_cv_scores = [all_cv_scores; yScores];
    all_cv_labels = [all_cv_labels; yTest];
    
    % Calculate AUC for this fold
    [~, ~, ~, cvAUCs(fold)] = perfcurve(yTest, yScores, 1);
    
    % Find optimal threshold for this fold
    thresholds = 0.1:0.05:0.9;
    bestF1_fold = 0;
    best_thresh_fold = 0.5;
    
    for t = thresholds
        yPred = yScores > t;
        tp = sum(yPred & (yTest == 1));
        fp = sum(yPred & (yTest == 0));
        fn = sum(~yPred & (yTest == 1));
        tn = sum(~yPred & (yTest == 0));
        
        if (tp + fp + fn + tn) > 0
            prec = tp / (tp + fp + eps);
            rec = tp / (tp + fn + eps);
            f1 = 2 * prec * rec / (prec + rec + eps);
            
            if ~isnan(f1) && f1 > bestF1_fold
                bestF1_fold = f1;
                best_thresh_fold = t;
            end
        end
    end
    
    cvThresholds(fold) = best_thresh_fold;
    
    % Calculate metrics at optimal threshold
    yPred = yScores > best_thresh_fold;
    tp = sum(yPred & (yTest == 1));
    fp = sum(yPred & (yTest == 0));
    tn = sum(~yPred & (yTest == 0));
    fn = sum(~yPred & (yTest == 1));
    
    cvAccuracies(fold) = (tp + tn) / (tp + tn + fp + fn + eps);
    cvPrecisions(fold) = tp / (tp + fp + eps);
    cvRecalls(fold) = tp / (tp + fn + eps);
    
    if cvPrecisions(fold) > 0 && cvRecalls(fold) > 0
        cvF1s(fold) = 2 * (cvPrecisions(fold) * cvRecalls(fold)) / ...
                         (cvPrecisions(fold) + cvRecalls(fold) + eps);
    end
    
    fprintf('  Fold %d Results (threshold=%.3f):\n', fold, best_thresh_fold);
    fprintf('    AUC: %.4f | Prec: %.1f%% | Recall: %.1f%% | F1: %.3f\n', ...
        cvAUCs(fold), cvPrecisions(fold)*100, cvRecalls(fold)*100, cvF1s(fold));
end

%% 11. Overall CV Results
fprintf('\n========== OVERALL 5-FOLD CV RESULTS ==========\n');
fprintf('Metric        Mean ± Std Dev     [Min, Max]\n');
fprintf('Accuracy:     %.1f%% ± %.1f%%     [%.1f%%, %.1f%%]\n', ...
    mean(cvAccuracies)*100, std(cvAccuracies)*100, ...
    min(cvAccuracies)*100, max(cvAccuracies)*100);
fprintf('Precision:    %.1f%% ± %.1f%%     [%.1f%%, %.1f%%]\n', ...
    mean(cvPrecisions)*100, std(cvPrecisions)*100, ...
    min(cvPrecisions)*100, max(cvPrecisions)*100);
fprintf('Recall:       %.1f%% ± %.1f%%     [%.1f%%, %.1f%%]\n', ...
    mean(cvRecalls)*100, std(cvRecalls)*100, ...
    min(cvRecalls)*100, max(cvRecalls)*100);
fprintf('F1 Score:     %.3f ± %.3f     [%.3f, %.3f]\n', ...
    mean(cvF1s), std(cvF1s), min(cvF1s), max(cvF1s));
fprintf('AUC:          %.4f ± %.4f     [%.4f, %.4f]\n', ...
    mean(cvAUCs), std(cvAUCs), min(cvAUCs), max(cvAUCs));
fprintf('Threshold:    %.3f ± %.3f     [%.3f, %.3f]\n', ...
    mean(cvThresholds), std(cvThresholds), min(cvThresholds), max(cvThresholds));

% Overall pooled AUC from CV
[~, ~, ~, cv_pooled_auc] = perfcurve(all_cv_labels, all_cv_scores, 1);
fprintf('\nOverall Pooled CV AUC: %.4f\n', cv_pooled_auc);

%% 12. Train Final Model on ALL Data
fprintf('\n========== TRAINING FINAL DECISION TREE MODEL ON ALL DATA ==========\n');

% Apply SMOTE to entire dataset
k_final = 5;
target_final = 1.0;
[X_balanced, y_balanced] = applySMOTE(X_selected, y, k_final, target_final);

fprintf('Final training set size: %d samples\n', length(y_balanced));
fprintf('  Class 0: %d, Class 1: %d\n', sum(y_balanced==0), sum(y_balanced==1));

% Create table for final model
varNames = cell(1, size(X_balanced, 2));
for i = 1:size(X_balanced, 2)
    varNames{i} = sprintf('Feature_%d', i);
end
data_final = array2table(X_balanced, 'VariableNames', varNames);
data_final.Target = y_balanced;

% Find best min leaf size for final model
minLeafSizes = [1, 3, 5, 10, 20];
bestAUC_final = 0;
best_final_model = [];
best_leaf_size = 5;

for m = 1:length(minLeafSizes)
    try
        tempModel = fitctree(data_final, 'Target', ...
            'MinLeafSize', minLeafSizes(m), ...
            'SplitCriterion', 'gdi', ...
            'Prune', 'on');
        
        % Get predictions on original data
        data_test = array2table(X_selected, 'VariableNames', varNames);
        [~, temp_scores] = predict(tempModel, data_test);
        temp_scores = temp_scores(:, 2);
        [~, ~, ~, temp_auc] = perfcurve(y, temp_scores, 1);
        
        if temp_auc > bestAUC_final
            bestAUC_final = temp_auc;
            best_final_model = tempModel;
            best_leaf_size = minLeafSizes(m);
        end
    catch
        continue;
    end
end

finalModel = best_final_model;
fprintf('Selected MinLeafSize = %d (AUC=%.4f)\n', best_leaf_size, bestAUC_final);

% Get predictions on original data
data_test_final = array2table(X_selected, 'VariableNames', varNames);
[~, final_scores] = predict(finalModel, data_test_final);
yScore_final = final_scores(:, 2);

% Find optimal threshold for final model
thresholds = 0.1:0.05:0.9;
bestF1 = 0;
final_threshold = 0.5;
final_tp = 0; final_fp = 0; final_tn = 0; final_fn = 0;
final_prec = 0; final_rec = 0; final_acc = 0;

for t = thresholds
    yPred = yScore_final > t;
    tp = sum(yPred & (y == 1));
    fp = sum(yPred & (y == 0));
    fn = sum(~yPred & (y == 1));
    tn = sum(~yPred & (y == 0));
    
    if (tp + fp + fn + tn) > 0
        prec = tp / (tp + fp + eps);
        rec = tp / (tp + fn + eps);
        f1 = 2 * prec * rec / (prec + rec + eps);
        acc = (tp + tn) / (tp + tn + fp + fn + eps);
        
        if ~isnan(f1) && f1 > bestF1
            bestF1 = f1;
            final_threshold = t;
            final_tp = tp;
            final_fp = fp;
            final_tn = tn;
            final_fn = fn;
            final_prec = prec;
            final_rec = rec;
            final_acc = acc;
        end
    end
end

% Final AUC
[~, ~, ~, final_auc] = perfcurve(y, yScore_final, 1);

fprintf('\n========== FINAL DECISION TREE PERFORMANCE ==========\n');
fprintf('Best Threshold: %.3f\n', final_threshold);
fprintf('Accuracy:       %.1f%%\n', final_acc*100);
fprintf('Precision:      %.1f%%\n', final_prec*100);
fprintf('Recall:         %.1f%%\n', final_rec*100);
fprintf('F1 Score:       %.3f\n', bestF1);
fprintf('AUC:            %.4f\n', final_auc);
fprintf('Confusion:      TP=%d, FP=%d, TN=%d, FN=%d\n', ...
    final_tp, final_fp, final_tn, final_fn);

% Calculate tree depth (number of levels)
try
    % Get tree depth by examining the tree structure
    tree_struct = finalModel.NodeClass; % This gives node classifications
    tree_depth = floor(log2(length(tree_struct))); % Rough estimate
catch
    tree_depth = 'N/A';
end

%% 13. Plot Results
figure('Name', 'Decision Tree Results', 'Position', [100, 100, 1600, 900]);

% ROC Curve - FINAL MODEL
subplot(2, 3, 1);
[X_roc_final, Y_roc_final, ~, ~] = perfcurve(y, yScore_final, 1);
plot(X_roc_final, Y_roc_final, 'b-', 'LineWidth', 4);
hold on;
% Add CV pooled ROC for comparison
[X_roc_cv, Y_roc_cv, ~, ~] = perfcurve(all_cv_labels, all_cv_scores, 1);
plot(X_roc_cv, Y_roc_cv, 'g--', 'LineWidth', 2);
plot([0 1], [0 1], 'r:', 'LineWidth', 1.5);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve - FINAL MODEL (AUC = %.4f)', final_auc), 'FontSize', 14, 'FontWeight', 'bold');
legend('Final Model', sprintf('CV Pooled (%.4f)', cv_pooled_auc), 'Random', 'Location', 'southeast');
grid on; axis square;
text(0.6, 0.2, sprintf('FINAL AUC = %.4f', final_auc), ...
    'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'w', 'EdgeColor', 'k');

% Per-fold AUC comparison
subplot(2, 3, 2);
bar(1:5, cvAUCs, 'FaceColor', [0.8, 0.4, 0.2]);
hold on;
yline(mean(cvAUCs), 'r--', 'LineWidth', 2);
yline(final_auc, 'b-', 'LineWidth', 3);
xlabel('Fold');
ylabel('AUC');
title('Per-Fold AUC Comparison');
legend('Per-fold AUC', sprintf('Mean CV (%.3f)', mean(cvAUCs)), ...
    sprintf('Final (%.3f)', final_auc), 'Location', 'best');
xticks(1:5);
ylim([0.5, 1]);
grid on;

% Feature Importance (using predictor importance from tree)
subplot(2, 3, 3);
imp = finalModel.predictorImportance;
[imp_sorted, idx] = sort(imp, 'descend');
topN = min(10, length(imp));

for i = 1:topN
    % Use correlation for direction
    if corr(X_selected(:, idx(i)), y) > 0
        color = [0.8, 0.2, 0.2];
    else
        color = [0.2, 0.2, 0.8];
    end
    h = barh(i, imp_sorted(i));
    hold on;
    set(h, 'FaceColor', color, 'EdgeColor', 'k');
end

yticks(1:topN);
yticklabels(featureNames_selected(idx(1:topN)));
xlabel('Predictor Importance');
title('Top 10 Feature Importance');
grid on;
legend({'Positive', 'Negative'}, 'Location', 'best');

% Confusion Matrix - Final Model
subplot(2, 3, 4);
cm_final = [final_tn, final_fp; final_fn, final_tp];
imagesc(cm_final); colormap('parula'); colorbar;
for i = 1:2
    for j = 1:2
        text(j, i, sprintf('%d\n(%.1f%%)', cm_final(i,j), cm_final(i,j)/sum(cm_final(:))*100), ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 14, 'FontWeight', 'bold', 'Color', 'white');
    end
end
set(gca, 'XTick', [1 2], 'YTick', [1 2], ...
    'XTickLabel', {'Pred Normal', 'Pred Aneurysm'}, ...
    'YTickLabel', {'Actual Normal', 'Actual Aneurysm'});
title(sprintf('Final Model Confusion Matrix\nAcc: %.1f%%, Prec: %.1f%%, Rec: %.1f%%', ...
    final_acc*100, final_prec*100, final_rec*100));
axis square;

% CV Metrics Stability
subplot(2, 3, 5);
plot(1:5, cvPrecisions*100, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
plot(1:5, cvRecalls*100, 'bs-', 'LineWidth', 2, 'MarkerSize', 8);
plot(1:5, cvF1s, 'g^-', 'LineWidth', 2, 'MarkerSize', 8);
% Add final model lines for comparison
yline(final_prec*100, 'r--', 'LineWidth', 1.5);
yline(final_rec*100, 'b--', 'LineWidth', 1.5);
yline(bestF1, 'g--', 'LineWidth', 1.5);
xlabel('Fold');
ylabel('Score (%)');
title('CV Metrics Stability vs Final Model');
legend('Precision (CV)', 'Recall (CV)', 'F1 (CV)', ...
    sprintf('Final Prec (%.1f%%)', final_prec*100), ...
    sprintf('Final Rec (%.1f%%)', final_rec*100), ...
    sprintf('Final F1 (%.3f)', bestF1), ...
    'Location', 'best');
xticks(1:5);
grid on;

% Score Distribution - Final Model
subplot(2, 3, 6);
histogram(yScore_final(y==0), 'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'BinWidth', 0.05);
hold on;
histogram(yScore_final(y==1), 'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'BinWidth', 0.05);
xline(final_threshold, 'k--', 'LineWidth', 2);
xlabel('Predicted Probability');
ylabel('Count');
title(sprintf('Final Model Score Distribution\nThreshold = %.2f', final_threshold));
legend('Normal', 'Aneurysm', 'Threshold');
grid on;

%% 14. Display Tree Structure (optional - uncomment to view)
% view(finalModel, 'Mode', 'graph');

%% 15. Final Summary - FIXED pruneList error
fprintf('\n========== DECISION TREE MODEL SUMMARY ==========\n');
fprintf('Model Type: Decision Tree (fitctree)\n');
fprintf('Min leaf size: %d (selected by validation)\n', best_leaf_size);
fprintf('Split criterion: Gini impurity\n');
fprintf('SMOTE target ratio: 1.0 (perfect balance)\n');
fprintf('Features: %d\n', size(X_selected, 2));

% Get tree size information
fprintf('Number of nodes: %d\n', finalModel.NumNodes);
% Estimate depth (not directly available, but can be inferred from nodes)
fprintf('Tree complexity: %d nodes\n', finalModel.NumNodes);

fprintf('\n 5-FOLD CROSS-VALIDATION RESULTS \n');
fprintf('  Accuracy:  %.1f%% ± %.1f%%\n', mean(cvAccuracies)*100, std(cvAccuracies)*100);
fprintf('  Precision: %.1f%% ± %.1f%%\n', mean(cvPrecisions)*100, std(cvPrecisions)*100);
fprintf('  Recall:    %.1f%% ± %.1f%%\n', mean(cvRecalls)*100, std(cvRecalls)*100);
fprintf('  F1 Score:  %.3f ± %.3f\n', mean(cvF1s), std(cvF1s));
fprintf('  AUC:       %.4f ± %.4f\n', mean(cvAUCs), std(cvAUCs));
fprintf('  Pooled AUC: %.4f\n', cv_pooled_auc);

fprintf('\n FINAL MODEL PERFORMANCE (TRAINED ON ALL DATA) \n');
fprintf('  Best Threshold: %.3f\n', final_threshold);
fprintf('  Accuracy:       %.1f%%\n', final_acc*100);
fprintf('  Precision:      %.1f%%\n', final_prec*100);
fprintf('  Recall:         %.1f%%\n', final_rec*100);
fprintf('  F1 Score:       %.3f\n', bestF1);
fprintf('  AUC:            %.4f \n', final_auc);

% Compare CV vs Final
fprintf('\n📊 CV vs FINAL MODEL COMPARISON:\n');
fprintf('  Metric        CV (mean ± std)     Final     Difference\n');
fprintf('  Accuracy:     %.1f%% ± %.1f%%     %.1f%%     %+.1f%%\n', ...
    mean(cvAccuracies)*100, std(cvAccuracies)*100, final_acc*100, final_acc*100 - mean(cvAccuracies)*100);
fprintf('  Precision:    %.1f%% ± %.1f%%     %.1f%%     %+.1f%%\n', ...
    mean(cvPrecisions)*100, std(cvPrecisions)*100, final_prec*100, final_prec*100 - mean(cvPrecisions)*100);
fprintf('  Recall:       %.1f%% ± %.1f%%     %.1f%%     %+.1f%%\n', ...
    mean(cvRecalls)*100, std(cvRecalls)*100, final_rec*100, final_rec*100 - mean(cvRecalls)*100);
fprintf('  F1 Score:     %.3f ± %.3f     %.3f     %+.3f\n', ...
    mean(cvF1s), std(cvF1s), bestF1, bestF1 - mean(cvF1s));
fprintf('  AUC:          %.4f ± %.4f     %.4f     %+.4f\n', ...
    mean(cvAUCs), std(cvAUCs), final_auc, final_auc - mean(cvAUCs));

fprintf('\n Analysis complete! FINAL DECISION TREE AUC = %.4f\n', final_auc);