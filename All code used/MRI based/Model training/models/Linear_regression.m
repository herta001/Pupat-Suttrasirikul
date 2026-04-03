%% Aneurysm MRI Dataset Analysis - Linear Regression Model with SMOTE (For Comparison)
% Clear workspace and command window
clear; clc; close all;

%% 1. Load and Prepare Data
% Assuming your data is in an Excel file named 'MRIdataset.xlsx'
filename = 'MRIdataset.xlsx';
data = readtable(filename);

% Display basic info
fprintf('Dataset loaded with %d rows and %d columns\n', height(data), width(data));
fprintf('Original class distribution:\n');
disp(table(data.PatientGroup));

%% 2. Create Binary Target Variable - NORMAL vs ANEURYSM ONLY
% Filter to keep only Normal and Aneurysm patients (exclude Hemorrhage)
normal_aneurysm_idx = strcmp(data.PatientGroup, 'Normal') | strcmp(data.PatientGroup, 'Aneurysm');
data_filtered = data(normal_aneurysm_idx, :);

% Create binary target: Normal (0) vs Aneurysm (1)
data_filtered.Target = double(strcmp(data_filtered.PatientGroup, 'Aneurysm'));

fprintf('\n========== NORMAL vs ANEURYSM CLASSIFICATION ==========\n');
fprintf('Filtered dataset: %d rows\n', height(data_filtered));
fprintf('Normal (0):   %d samples\n', sum(data_filtered.Target == 0));
fprintf('Aneurysm (1): %d samples\n', sum(data_filtered.Target == 1));

%% 3. Select Features for Analysis
% Based on medical literature and feature importance, selecting key features
selectedFeatures = {
    'num_vessels', ...              % Number of vessels
    'vessel_density', ...            % Vessel density
    'mean_width', ...                % Mean vessel width
    'std_width', ...                  % Standard deviation of width
    'mean_aspect_ratio', ...          % Mean aspect ratio
    'branch_point_density', ...       % Branch point density
    'end_point_density', ...          % End point density
    'mean_circularity', ...           % Mean circularity
    'intensity_range', ...            % Intensity range
    'intensity_std', ...               % Intensity standard deviation
    'large_vessel_count', ...         % Count of large vessels
    'small_vessel_count', ...         % Count of small vessels
    'total_vessel_length', ...        % Total vessel length
    'vessel_complexity', ...          % Vessel complexity
    'glc_contrast', ...               % GLCM contrast
    'glc_homogeneity'                 % GLCM homogeneity
};

% Check which features exist in the dataset
availableFeatures = {};
for i = 1:length(selectedFeatures)
    if ismember(selectedFeatures{i}, data_filtered.Properties.VariableNames)
        availableFeatures{end+1} = selectedFeatures{i};
    else
        fprintf('Warning: Feature "%s" not found in dataset\n', selectedFeatures{i});
    end
end

fprintf('\nUsing %d features for analysis\n', length(availableFeatures));

%% 4. Prepare Feature Matrix and Handle Missing/Infinite Values
X = [];
featureNames = {};

for i = 1:length(availableFeatures)
    featureData = data_filtered.(availableFeatures{i});
    
    % Handle infinite values
    featureData(isinf(featureData)) = NaN;
    
    X = [X, featureData];
    featureNames{end+1} = availableFeatures{i};
end

y = data_filtered.Target;

%% 5. Remove Rows with NaN or Invalid Values
validRows = all(~isnan(X), 2) & all(isfinite(X), 2);
X = X(validRows, :);
y = y(validRows);

fprintf('\nAfter removing invalid data: %d samples remain\n', length(y));
fprintf('  Normal:   %d samples (%.1f%%)\n', sum(y == 0), sum(y==0)/length(y)*100);
fprintf('  Aneurysm: %d samples (%.1f%%)\n', sum(y == 1), sum(y==1)/length(y)*100);

%% 6. Normalize Features
X_normalized = normalize(X);

%% 7. SMOTE Implementation Function (Optimized)
function [X_smote, y_smote] = applySMOTE(X, y, k, targetRatio)
    % Separate majority and minority classes
    X_majority = X(y == 0, :);
    X_minority = X(y == 1, :);
    y_majority = y(y == 0);
    
    n_majority = size(X_majority, 1);
    n_minority = size(X_minority, 1);
    
    % Calculate how many synthetic samples to create
    target_minority = round(n_majority * targetRatio);
    n_synthetic = max(0, target_minority - n_minority);
    
    fprintf('SMOTE: Creating %d synthetic minority samples (target ratio=%.2f)\n', n_synthetic, targetRatio);
    
    if n_synthetic <= 0
        X_smote = X;
        y_smote = y;
        return;
    end
    
    % Pre-allocate synthetic samples
    n_features = size(X, 2);
    synthetic_samples = zeros(n_synthetic, n_features);
    
    % Pre-compute all nearest neighbors for minority samples (optimization)
    all_neighbors = zeros(n_minority, k);
    for i = 1:n_minority
        distances = sum((X_minority - X_minority(i,:)).^2, 2);
        [~, neighbor_idx] = mink(distances, k+1);
        all_neighbors(i, :) = neighbor_idx(2:end);
    end
    
    % Generate synthetic samples
    for i = 1:n_synthetic
        idx = randi(n_minority);
        base_sample = X_minority(idx, :);
        
        neighbor = all_neighbors(idx, randi(k));
        
        gap = rand(1, n_features);
        synthetic_samples(i, :) = base_sample + gap .* (X_minority(neighbor, :) - base_sample);
    end
    
    X_smote = [X_majority; X_minority; synthetic_samples];
    y_smote = [y_majority; y(y == 1); ones(n_synthetic, 1)];
end

%% 8. Set Up 5-Fold Cross-Validation
rng(42); % For reproducibility
cv = cvpartition(y, 'KFold', 5);

fprintf('\n========== 5-FOLD CROSS-VALIDATION WITH LINEAR REGRESSION (FOR COMPARISON) ==========\n');

% Initialize arrays to store results from each fold
foldAccuracies = zeros(5, 1);
foldPrecisions = zeros(5, 1);
foldRecalls = zeros(5, 1);
foldF1s = zeros(5, 1);
foldSpecificities = zeros(5, 1);
foldAUCs = zeros(5, 1);
foldThresholds = zeros(5, 1);
allScores = zeros(length(y), 1);
allTrueLabels = zeros(length(y), 1);
scoreIdx = 1;

%% 9. Perform 5-Fold Cross-Validation with SMOTE (Linear Regression)
for fold = 1:5
    fprintf('\n--- Fold %d of 5 ---\n', fold);
    
    % Get training and testing indices for this fold
    trainIdx = cv.training(fold);
    testIdx = cv.test(fold);
    
    XTrain_raw = X_normalized(trainIdx, :);
    yTrain_raw = y(trainIdx);
    XTest = X_normalized(testIdx, :);
    yTest = y(testIdx);
    
    % Apply SMOTE to training data
    k_neighbors = 5;
    target_ratio = 0.4;
    
    nNormal_train = sum(yTrain_raw == 0);
    nAneurysm_train = sum(yTrain_raw == 1);
    
    fprintf('Before SMOTE - Training: %d samples (Normal: %d, Aneurysm: %d)\n', ...
        length(yTrain_raw), nNormal_train, nAneurysm_train);
    
    [XTrain, yTrain] = applySMOTE(XTrain_raw, yTrain_raw, k_neighbors, target_ratio);
    
    fprintf('After SMOTE - Training: %d samples (Normal: %d, Aneurysm: %d)\n', ...
        length(yTrain), sum(yTrain==0), sum(yTrain==1));
    fprintf('Testing set: %d samples (Normal: %d, Aneurysm: %d)\n', ...
        length(yTest), sum(yTest==0), sum(yTest==1));
    
    % Calculate sample weights (optimized - use repelem instead of loops)
    weightNormal = 1.2;
    weightAneurysm = 1.0;
    
    % Create weight multipliers for each sample
    weightMultipliers = zeros(length(yTrain), 1);
    weightMultipliers(yTrain == 0) = round(weightNormal * 10);
    weightMultipliers(yTrain == 1) = round(weightAneurysm * 10);
    
    % Use repelem for faster duplication
    XTrain_weighted = repelem(XTrain, weightMultipliers, 1);
    yTrain_weighted = repelem(yTrain, weightMultipliers, 1);
    
    fprintf('After weighting - Training: %d samples\n', length(yTrain_weighted));
    
    % Train Linear Regression model (faster with matrix operations)
    X_with_intercept = [ones(size(XTrain_weighted, 1), 1), XTrain_weighted];
    beta = (X_with_intercept' * X_with_intercept) \ (X_with_intercept' * yTrain_weighted);
    
    % Create model structure for consistency
    mdl = struct();
    mdl.Coefficients.Estimate = beta;
    
    % Store model
    foldModels{fold} = mdl;
    
    % Get predictions - use matrix multiplication for speed
    XTest_with_intercept = [ones(size(XTest, 1), 1), XTest];
    yScore = XTest_with_intercept * beta;
    
    % Clip predictions to [0, 1] range
    yScore = max(0, min(1, yScore));
    
    % Store scores for overall AUC calculation
    nTest = length(yTest);
    allScores(scoreIdx:scoreIdx+nTest-1) = yScore;
    allTrueLabels(scoreIdx:scoreIdx+nTest-1) = yTest;
    scoreIdx = scoreIdx + nTest;
    
    % Find optimal threshold (optimized)
    thresholds = 0.5:0.025:0.9;
    bestPrecision = 0;
    bestThreshold = 0.675;
    
    % Get training predictions
    XTrain_weighted_with_intercept = [ones(size(XTrain_weighted, 1), 1), XTrain_weighted];
    yScore_train = XTrain_weighted_with_intercept * beta;
    yScore_train = max(0, min(1, yScore_train));
    
    % Vectorized threshold optimization
    for t = thresholds
        pred_train = yScore_train > t;
        tp_train = sum(pred_train & (yTrain_weighted == 1));
        fp_train = sum(pred_train & (yTrain_weighted == 0));
        fn_train = sum(~pred_train & (yTrain_weighted == 1));
        
        if tp_train > 0
            prec_train = tp_train / (tp_train + fp_train);
            rec_train = tp_train / (tp_train + fn_train);
            
            if rec_train >= 0.35 && prec_train > bestPrecision && ~isnan(prec_train)
                bestPrecision = prec_train;
                bestThreshold = t;
            end
        end
    end
    
    optThreshold = bestThreshold;
    foldThresholds(fold) = optThreshold;
    
    % Make predictions using optimal threshold
    yPred = yScore > optThreshold;
    
    % Calculate confusion matrix for this fold
    tp = sum(yPred & (yTest == 1));
    fp = sum(yPred & (yTest == 0));
    tn = sum(~yPred & (yTest == 0));
    fn = sum(~yPred & (yTest == 1));
    
    % Calculate metrics for this fold
    total = tp + tn + fp + fn;
    foldAccuracies(fold) = (tp + tn) / total;
    foldPrecisions(fold) = tp / (tp + fp + eps);
    foldRecalls(fold) = tp / (tp + fn + eps);
    foldSpecificities(fold) = tn / (tn + fp + eps);
    
    if foldPrecisions(fold) > 0 && foldRecalls(fold) > 0
        foldF1s(fold) = 2 * (foldPrecisions(fold) * foldRecalls(fold)) / ...
                           (foldPrecisions(fold) + foldRecalls(fold));
    end
    
    % Calculate AUC for this fold
    try
        [~, ~, ~, foldAUCs(fold)] = perfcurve(yTest, yScore, 1);
    catch
        foldAUCs(fold) = 0.5;
    end
    
    % Display fold results
    fprintf('Fold %d Results (threshold = %.3f):\n', fold, optThreshold);
    fprintf('  Accuracy:    %.4f (%.1f%%)\n', foldAccuracies(fold), foldAccuracies(fold)*100);
    fprintf('  Precision:   %.4f (%.1f%%)\n', foldPrecisions(fold), foldPrecisions(fold)*100);
    fprintf('  Recall:      %.4f (%.1f%%)\n', foldRecalls(fold), foldRecalls(fold)*100);
    fprintf('  F1 Score:    %.4f\n', foldF1s(fold));
    fprintf('  AUC:         %.4f\n', foldAUCs(fold));
    fprintf('  Confusion:   TP=%d, FP=%d, TN=%d, FN=%d\n', tp, fp, tn, fn);
end

% Trim unused arrays
allScores = allScores(1:scoreIdx-1);
allTrueLabels = allTrueLabels(1:scoreIdx-1);

%% 10. Calculate Overall Cross-Validated Performance
fprintf('\n========== OVERALL 5-FOLD CV RESULTS (LINEAR REGRESSION) ==========\n');
fprintf('Metric        Mean ± Std Dev\n');
fprintf('Accuracy:     %.4f ± %.4f (%.1f%% ± %.1f%%)\n', ...
    mean(foldAccuracies), std(foldAccuracies), ...
    mean(foldAccuracies)*100, std(foldAccuracies)*100);
fprintf('Precision:    %.4f ± %.4f (%.1f%% ± %.1f%%)\n', ...
    mean(foldPrecisions), std(foldPrecisions), ...
    mean(foldPrecisions)*100, std(foldPrecisions)*100);
fprintf('Recall:       %.4f ± %.4f (%.1f%% ± %.1f%%)\n', ...
    mean(foldRecalls), std(foldRecalls), ...
    mean(foldRecalls)*100, std(foldRecalls)*100);
fprintf('F1 Score:     %.4f ± %.4f\n', mean(foldF1s), std(foldF1s));
fprintf('Specificity:  %.4f ± %.4f (%.1f%% ± %.1f%%)\n', ...
    mean(foldSpecificities), std(foldSpecificities), ...
    mean(foldSpecificities)*100, std(foldSpecificities)*100);
fprintf('AUC:          %.4f ± %.4f\n', mean(foldAUCs), std(foldAUCs));
fprintf('Threshold:    %.4f ± %.4f\n', mean(foldThresholds), std(foldThresholds));

% Calculate overall AUC using all predictions
try
    [X_roc, Y_roc, ~, overallAUC] = perfcurve(allTrueLabels, allScores, 1);
catch
    X_roc = [0, 1];
    Y_roc = [0, 1];
    overallAUC = 0.5;
end
fprintf('\nOverall Pooled AUC: %.4f\n', overallAUC);

%% 11. Train Final Model on All Data with SMOTE (Linear Regression)
fprintf('\n========== TRAINING FINAL MODEL ON ALL DATA (LINEAR REGRESSION) ==========\n');

% Apply SMOTE to entire dataset
k_neighbors_final = 5;
target_ratio_final = 0.4;

nNormal_total = sum(y == 0);
nAneurysm_total = sum(y == 1);

fprintf('Before SMOTE - Full dataset: %d samples (Normal: %d, Aneurysm: %d)\n', ...
    length(y), nNormal_total, nAneurysm_total);

[X_smote_full, y_smote_full] = applySMOTE(X_normalized, y, k_neighbors_final, target_ratio_final);

fprintf('After SMOTE - Full dataset: %d samples (Normal: %d, Aneurysm: %d)\n', ...
    length(y_smote_full), sum(y_smote_full==0), sum(y_smote_full==1));

% Apply weighting by sample duplication (optimized)
weightNormal_smote = 1.2;
weightAneurysm_smote = 1.0;

weightMultipliers_full = zeros(length(y_smote_full), 1);
weightMultipliers_full(y_smote_full == 0) = round(weightNormal_smote * 10);
weightMultipliers_full(y_smote_full == 1) = round(weightAneurysm_smote * 10);

X_final_weighted = repelem(X_smote_full, weightMultipliers_full, 1);
y_final_weighted = repelem(y_smote_full, weightMultipliers_full, 1);

fprintf('After weighting - Final training: %d samples\n', length(y_final_weighted));

% Train final linear regression model (using matrix operations for speed)
X_final_with_intercept = [ones(size(X_final_weighted, 1), 1), X_final_weighted];
beta_final = (X_final_with_intercept' * X_final_with_intercept) \ (X_final_with_intercept' * y_final_weighted);

finalModel = struct();
finalModel.Coefficients.Estimate = beta_final;

% Get predictions on ORIGINAL data
X_orig_with_intercept = [ones(size(X_normalized, 1), 1), X_normalized];
yScore_final = X_orig_with_intercept * beta_final;
yScore_final = max(0, min(1, yScore_final));

% Find threshold that maximizes precision
thresholds = 0.5:0.025:0.9;
bestPrecision = 0;
final_threshold = 0.675;
best_precision_val = 0;
best_recall_val = 0;

for t = thresholds
    pred_final = yScore_final > t;
    tp_final = sum(pred_final & (y == 1));
    fp_final = sum(pred_final & (y == 0));
    fn_final = sum(~pred_final & (y == 1));
    
    if tp_final > 0
        prec_final = tp_final / (tp_final + fp_final + eps);
        rec_final = tp_final / (tp_final + fn_final + eps);
        
        if rec_final >= 0.35 && prec_final > bestPrecision && ~isnan(prec_final)
            bestPrecision = prec_final;
            final_threshold = t;
            best_precision_val = prec_final;
            best_recall_val = rec_final;
        end
    end
end

% Make final predictions using optimized threshold
yPred_final = yScore_final > final_threshold;

% Calculate final confusion matrix on ORIGINAL data
tp_final = sum(yPred_final & (y == 1));
fp_final = sum(yPred_final & (y == 0));
tn_final = sum(~yPred_final & (y == 0));
fn_final = sum(~yPred_final & (y == 1));

% Calculate final metrics on ORIGINAL data
final_accuracy = (tp_final + tn_final) / (tp_final + tn_final + fp_final + fn_final + eps);
final_precision = tp_final / (tp_final + fp_final + eps);
final_recall = tp_final / (tp_final + fn_final + eps);
final_specificity = tn_final / (tn_final + fp_final + eps);

if final_precision > 0 && final_recall > 0
    final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall);
else
    final_f1 = 0;
end

fprintf('\nFinal Model Performance on ORIGINAL Data (threshold = %.3f):\n', final_threshold);
fprintf('  Accuracy:    %.4f (%.1f%%)\n', final_accuracy, final_accuracy*100);
fprintf('  Precision:   %.4f (%.1f%%)\n', final_precision, final_precision*100);
fprintf('  Recall:      %.4f (%.1f%%)\n', final_recall, final_recall*100);
fprintf('  F1 Score:    %.4f\n', final_f1);
fprintf('  Specificity: %.4f (%.1f%%)\n', final_specificity, final_specificity*100);
fprintf('  Confusion:   TP=%d, FP=%d, TN=%d, FN=%d\n', tp_final, fp_final, tn_final, fn_final);

%% 12. Plot Results
figure('Name', 'Normal vs Aneurysm Classification (Linear Regression - For Comparison)', 'Position', [100, 100, 1400, 500]);

% Subplot 1: ROC Curve
subplot(1, 3, 1);
plot(X_roc, Y_roc, 'b-', 'LineWidth', 3);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);
xlabel('False Positive Rate (1 - Specificity)', 'FontSize', 12);
ylabel('True Positive Rate (Sensitivity)', 'FontSize', 12);
title(sprintf('ROC Curve (AUC = %.3f) - Linear Regression', overallAUC), 'FontSize', 14);
legend('Linear Regression with SMOTE', 'Random Classifier', 'Location', 'southeast');
grid on;
axis square;

% Subplot 2: Feature Importance (Coefficients)
subplot(1, 3, 2);
coeff = beta_final(2:end); % Skip intercept
[~, idx] = sort(abs(coeff), 'descend');
topN = min(10, length(coeff));

% Create bar plot efficiently
for i = 1:topN
    if coeff(idx(i)) > 0
        color = [1, 0.6, 0.6];
    else
        color = [0.6, 0.6, 1];
    end
    h = barh(i, coeff(idx(i)));
    hold on;
    set(h, 'FaceColor', color, 'EdgeColor', 'k');
end

yticks(1:topN);
yticklabels(featureNames(idx(1:topN)));
xlabel('Coefficient Value', 'FontSize', 12);
title('Top 10 Feature Coefficients', 'FontSize', 14);
grid on;
legend({'Positive', 'Negative'}, 'Location', 'best', 'FontSize', 9);

% Subplot 3: Confusion Matrix
subplot(1, 3, 3);
cm_final = [tn_final, fp_final; fn_final, tp_final];
imagesc(cm_final);
colormap('parula');
colorbar;

[x_grid, y_grid] = meshgrid(1:2);
for i = 1:numel(x_grid)
    text(x_grid(i), y_grid(i), num2str(cm_final(i)), ...
        'HorizontalAlignment', 'center', ...
        'FontSize', 20, 'FontWeight', 'bold', 'Color', 'white');
end

set(gca, 'XTick', [1, 2], 'YTick', [1, 2], ...
    'XTickLabel', {'Predicted Normal', 'Predicted Aneurysm'}, ...
    'YTickLabel', {'Actual Normal', 'Actual Aneurysm'}, ...
    'FontSize', 11);
title(sprintf('Confusion Matrix - Linear Regression\nAccuracy: %.1f%% | Precision: %.1f%%', ...
    final_accuracy*100, final_precision*100), 'FontSize', 14);
axis square;

text(0.5, -0.3, sprintf('Recall: %.1f%% | F1: %.3f', ...
    final_recall*100, final_f1), ...
    'HorizontalAlignment', 'center', 'FontSize', 11, 'Units', 'normalized');

%% 13. Model Summary
fprintf('\n========== MODEL SUMMARY: LINEAR REGRESSION with SMOTE (FOR COMPARISON) ==========\n');
fprintf('Model Type: Linear Regression with SMOTE + 5-Fold CV\n');
fprintf('SMOTE parameters: k=%d, target ratio=%.2f\n', k_neighbors_final, target_ratio_final);
fprintf('Features used: %d\n', length(availableFeatures));
fprintf('Original samples: %d (Normal: %d, Aneurysm: %d)\n', length(y), nNormal_total, nAneurysm_total);
fprintf('SMOTE-enhanced samples: %d (Normal: %d, Aneurysm: %d)\n', ...
    length(y_smote_full), sum(y_smote_full==0), sum(y_smote_full==1));
fprintf('\n5-Fold CV Performance (Mean ± Std):\n');
fprintf('  Accuracy:  %.1f%% ± %.1f%%\n', mean(foldAccuracies)*100, std(foldAccuracies)*100);
fprintf('  Precision: %.1f%% ± %.1f%%\n', mean(foldPrecisions)*100, std(foldPrecisions)*100);
fprintf('  Recall:    %.1f%% ± %.1f%%\n', mean(foldRecalls)*100, std(foldRecalls)*100);
fprintf('  F1 Score:  %.3f ± %.3f\n', mean(foldF1s), std(foldF1s));
fprintf('  AUC:       %.3f ± %.3f\n', mean(foldAUCs), std(foldAUCs));

fprintf('\nFinal Model Performance (original data, threshold = %.3f):\n', final_threshold);
fprintf('  Accuracy:  %.1f%%\n', final_accuracy*100);
fprintf('  Precision: %.1f%%\n', final_precision*100);
fprintf('  Recall:    %.1f%%\n', final_recall*100);
fprintf('  F1 Score:  %.3f\n', final_f1);

fprintf('\nAnalysis complete! Results displayed in figure.\n');