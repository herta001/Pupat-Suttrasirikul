%% Aneurysm MRI Dataset Analysis - Logistic Regression with SMOTE (Optimized for Precision)
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

%% 7. SMOTE Implementation Function (Improved for Precision)
function [X_smote, y_smote] = applySMOTE(X, y, k, targetRatio)
    % X: feature matrix
    % y: labels (0 for majority, 1 for minority)
    % k: number of nearest neighbors
    % targetRatio: desired minority/majority ratio (e.g., 0.3 = 30% of majority)
    
    % Separate majority and minority classes
    X_majority = X(y == 0, :);
    X_minority = X(y == 1, :);
    y_majority = y(y == 0);
    y_minority = y(y == 1);
    
    n_majority = size(X_majority, 1);
    n_minority = size(X_minority, 1);
    
    % Calculate how many synthetic samples to create
    target_minority = round(n_majority * targetRatio);
    n_synthetic = max(0, target_minority - n_minority);
    
    fprintf('SMOTE: Creating %d synthetic minority samples (target ratio=%.2f)\n', n_synthetic, targetRatio);
    
    if n_synthetic <= 0
        % Already at or above target ratio
        X_smote = X;
        y_smote = y;
        return;
    end
    
    % For each synthetic sample to create
    synthetic_samples = [];
    
    for i = 1:n_synthetic
        % Randomly select a minority sample
        idx = randi(n_minority);
        base_sample = X_minority(idx, :);
        
        % Find k nearest neighbors among minority class
        distances = pdist2(base_sample, X_minority);
        [~, neighbor_idx] = sort(distances);
        
        % Select a random neighbor from the k nearest (excluding itself)
        max_neighbor = min(k, length(neighbor_idx)-1);
        if max_neighbor < 1
            continue;
        end
        neighbor = neighbor_idx(randi(max_neighbor) + 1);
        
        % Create synthetic sample by interpolation with random gap
        % Use smaller random gap for more conservative synthetic samples
        gap = rand(1, size(X, 2)) * 0.5; % Reduced from 1.0 to 0.5 for less variation
        synthetic = base_sample + gap .* (X_minority(neighbor, :) - base_sample);
        synthetic_samples = [synthetic_samples; synthetic];
    end
    
    % Combine original data with synthetic samples
    X_smote = [X_majority; X_minority; synthetic_samples];
    y_smote = [y_majority; y_minority; ones(n_synthetic, 1)];
end

%% 8. Set Up 5-Fold Cross-Validation with SMOTE
rng(42); % For reproducibility
cv = cvpartition(y, 'KFold', 5);

fprintf('\n========== 5-FOLD CROSS-VALIDATION WITH SMOTE (OPTIMIZED FOR PRECISION) ==========\n');

% Initialize arrays to store results from each fold
foldAccuracies = zeros(5, 1);
foldPrecisions = zeros(5, 1);
foldRecalls = zeros(5, 1);
foldF1s = zeros(5, 1);
foldSpecificities = zeros(5, 1);
foldAUCs = zeros(5, 1);
foldThresholds = zeros(5, 1);
foldModels = cell(5, 1);
allScores = [];
allTrueLabels = [];

%% 9. Perform 5-Fold Cross-Validation with SMOTE (Optimized for Precision)
for fold = 1:5
    fprintf('\n--- Fold %d of 5 ---\n', fold);
    
    % Get training and testing indices for this fold
    trainIdx = cv.training(fold);
    testIdx = cv.test(fold);
    
    XTrain_raw = X_normalized(trainIdx, :);
    yTrain_raw = y(trainIdx);
    XTest = X_normalized(testIdx, :);
    yTest = y(testIdx);
    
    % Apply SMOTE to training data - target minority at 40% of majority (less balancing)
    k_neighbors = 5; % Fewer neighbors for more conservative synthetic samples
    target_ratio = 0.4; % Target minority = 40% of majority (less than before)
    
    nNormal_train = sum(yTrain_raw == 0);
    nAneurysm_train = sum(yTrain_raw == 1);
    
    fprintf('Before SMOTE - Training: %d samples (Normal: %d, Aneurysm: %d)\n', ...
        length(yTrain_raw), nNormal_train, nAneurysm_train);
    
    [XTrain, yTrain] = applySMOTE(XTrain_raw, yTrain_raw, k_neighbors, target_ratio);
    
    fprintf('After SMOTE - Training: %d samples (Normal: %d, Aneurysm: %d)\n', ...
        length(yTrain), sum(yTrain==0), sum(yTrain==1));
    fprintf('Testing set: %d samples (Normal: %d, Aneurysm: %d)\n', ...
        length(yTest), sum(yTest==0), sum(yTest==1));
    
    % Calculate class weights - emphasize normal class to improve precision
    nNormal = sum(yTrain == 0);
    nAneurysm = sum(yTrain == 1);
    
    % Higher weight on normal class to reduce false positives
    weightNormal = 1.2; % Increased from 1.0
    weightAneurysm = 1.0; % Lower weight on aneurysm class
    
    % Create weight vector for each training sample
    sampleWeights = zeros(size(yTrain));
    sampleWeights(yTrain == 0) = weightNormal;
    sampleWeights(yTrain == 1) = weightAneurysm;
    
    fprintf('Class weights - Normal: %.2f, Aneurysm: %.2f\n', weightNormal, weightAneurysm);
    
    % Train model for this fold with stronger regularization
    if length(availableFeatures) > 10
        % Slightly higher lambda for more conservative model
        lambda = 0.002; % Increased from 0.0005
        
        mdl = fitclinear(XTrain, yTrain, ...
            'Learner', 'logistic', ...
            'Regularization', 'ridge', ...
            'Lambda', lambda, ...
            'Solver', 'sgd', ...
            'Weights', sampleWeights, ...
            'Verbose', 0);
    else
        mdl = fitglm(XTrain, yTrain, 'Distribution', 'binomial', 'Link', 'logit', ...
            'Weights', sampleWeights);
    end
    
    % Store model
    foldModels{fold} = mdl;
    
    % Get predictions and scores
    if exist('mdl', 'var') && isa(mdl, 'ClassificationLinear')
        [~, scores] = predict(mdl, XTest);
        yScore = scores(:, 2); % Probability of class 1 (Aneurysm)
        
        % Also get training scores to find optimal threshold
        [~, scores_train] = predict(mdl, XTrain);
        yScore_train = scores_train(:, 2);
    elseif exist('mdl', 'var') && isa(mdl, 'GeneralizedLinearModel')
        yScore = predict(mdl, XTest);
        yScore_train = predict(mdl, XTrain);
    else
        error('Model not properly trained');
    end
    
    % Find threshold that maximizes precision with minimum recall of 40%
    try
        % Higher threshold range for better precision
        thresholds = 0.5:0.025:0.9;
        bestPrecision = 0;
        bestThreshold = 0.675; % Default based on your results
        
        for t = thresholds
            pred_train = double(yScore_train > t);
            tp_train = sum(pred_train == 1 & yTrain == 1);
            fp_train = sum(pred_train == 1 & yTrain == 0);
            fn_train = sum(pred_train == 0 & yTrain == 1);
            
            if tp_train > 0
                prec_train = tp_train / (tp_train + fp_train);
                rec_train = tp_train / (tp_train + fn_train);
                
                % Only consider thresholds that maintain minimum recall
                if rec_train >= 0.35 % Lower recall threshold to allow higher precision
                    if prec_train > bestPrecision && ~isnan(prec_train)
                        bestPrecision = prec_train;
                        bestThreshold = t;
                    end
                end
            end
        end
        
        optThreshold = bestThreshold;
        
    catch
        optThreshold = 0.675; % Default based on your results
    end
    
    foldThresholds(fold) = optThreshold;
    
    % Make predictions using optimal threshold
    yPred = double(yScore > optThreshold);
    
    % Store scores for overall AUC calculation
    allScores = [allScores; yScore];
    allTrueLabels = [allTrueLabels; yTest];
    
    % Calculate confusion matrix for this fold
    cm = confusionmat(yTest, yPred);
    tn = cm(1,1); fp = cm(1,2);
    fn = cm(2,1); tp = cm(2,2);
    
    % Calculate metrics for this fold
    foldAccuracies(fold) = (tp + tn) / (tp + tn + fp + fn);
    
    % Handle division by zero for precision
    if (tp + fp) > 0
        foldPrecisions(fold) = tp / (tp + fp);
    else
        foldPrecisions(fold) = 0;
    end
    
    % Handle division by zero for recall
    if (tp + fn) > 0
        foldRecalls(fold) = tp / (tp + fn);
    else
        foldRecalls(fold) = 0;
    end
    
    % Calculate F1
    if foldPrecisions(fold) > 0 && foldRecalls(fold) > 0
        foldF1s(fold) = 2 * (foldPrecisions(fold) * foldRecalls(fold)) / ...
                           (foldPrecisions(fold) + foldRecalls(fold));
    else
        foldF1s(fold) = 0;
    end
    
    % Handle division by zero for specificity
    if (tn + fp) > 0
        foldSpecificities(fold) = tn / (tn + fp);
    else
        foldSpecificities(fold) = 0;
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

%% 10. Calculate Overall Cross-Validated Performance
fprintf('\n========== OVERALL 5-FOLD CV RESULTS (OPTIMIZED FOR PRECISION) ==========\n');
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

%% 11. Train Final Model on All Data with SMOTE (Optimized for Precision)
fprintf('\n========== TRAINING FINAL MODEL ON ALL DATA (OPTIMIZED FOR PRECISION) ==========\n');

% Apply SMOTE to entire dataset - more conservative
k_neighbors_final = 5;
target_ratio_final = 0.4; % Target minority at 40% of majority

nNormal_total = sum(y == 0);
nAneurysm_total = sum(y == 1);

fprintf('Before SMOTE - Full dataset: %d samples (Normal: %d, Aneurysm: %d)\n', ...
    length(y), nNormal_total, nAneurysm_total);

[X_smote_full, y_smote_full] = applySMOTE(X_normalized, y, k_neighbors_final, target_ratio_final);

fprintf('After SMOTE - Full dataset: %d samples (Normal: %d, Aneurysm: %d)\n', ...
    length(y_smote_full), sum(y_smote_full==0), sum(y_smote_full==1));

% Class weights favoring normal class
weightNormal_smote = 1.2;
weightAneurysm_smote = 1.0;

% Create weight vector
sampleWeights_smote = zeros(size(y_smote_full));
sampleWeights_smote(y_smote_full == 0) = weightNormal_smote;
sampleWeights_smote(y_smote_full == 1) = weightAneurysm_smote;

fprintf('Class weights - Normal: %.2f, Aneurysm: %.2f\n', weightNormal_smote, weightAneurysm_smote);

% Train final model on SMOTE-enhanced data with stronger regularization
if length(availableFeatures) > 10
    lambda = 0.002; % Stronger regularization for more conservative model
    finalModel = fitclinear(X_smote_full, y_smote_full, ...
        'Learner', 'logistic', ...
        'Regularization', 'ridge', ...
        'Lambda', lambda, ...
        'Solver', 'sgd', ...
        'Weights', sampleWeights_smote, ...
        'Verbose', 0);
else
    finalModel = fitglm(X_smote_full, y_smote_full, 'Distribution', 'binomial', 'Link', 'logit', ...
        'Weights', sampleWeights_smote);
end

% Get predictions on ORIGINAL data
if isa(finalModel, 'ClassificationLinear')
    [~, scores_final] = predict(finalModel, X_normalized);
    yScore_final = scores_final(:, 2);
else
    yScore_final = predict(finalModel, X_normalized);
end

% Find threshold that maximizes precision with minimum recall constraint
try
    thresholds = 0.5:0.025:0.9;
    bestPrecision = 0;
    final_threshold = 0.675;
    best_precision_val = 0;
    best_recall_val = 0;
    
    for t = thresholds
        pred_final = double(yScore_final > t);
        tp_final = sum(pred_final == 1 & y == 1);
        fp_final = sum(pred_final == 1 & y == 0);
        fn_final = sum(pred_final == 0 & y == 1);
        
        if tp_final > 0
            prec_final = tp_final / (tp_final + fp_final);
            rec_final = tp_final / (tp_final + fn_final);
            
            % Only consider thresholds that maintain minimum recall
            if rec_final >= 0.35 % Lower recall threshold for higher precision
                if prec_final > bestPrecision && ~isnan(prec_final)
                    bestPrecision = prec_final;
                    final_threshold = t;
                    best_precision_val = prec_final;
                    best_recall_val = rec_final;
                end
            end
        end
    end
catch
    final_threshold = mean(foldThresholds);
end

% Make final predictions using optimized threshold
yPred_final = double(yScore_final > final_threshold);

% Calculate final confusion matrix on ORIGINAL data
cm_final = confusionmat(y, yPred_final);
tn_final = cm_final(1,1); fp_final = cm_final(1,2);
fn_final = cm_final(2,1); tp_final = cm_final(2,2);

% Calculate final metrics on ORIGINAL data
final_accuracy = (tp_final + tn_final) / (tp_final + tn_final + fp_final + fn_final);

if (tp_final + fp_final) > 0
    final_precision = tp_final / (tp_final + fp_final);
else
    final_precision = 0;
end

if (tp_final + fn_final) > 0
    final_recall = tp_final / (tp_final + fn_final);
else
    final_recall = 0;
end

if final_precision > 0 && final_recall > 0
    final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall);
else
    final_f1 = 0;
end

if (tn_final + fp_final) > 0
    final_specificity = tn_final / (tn_final + fp_final);
else
    final_specificity = 0;
end

fprintf('\nFinal Model Performance on ORIGINAL Data (threshold = %.3f):\n', final_threshold);
fprintf('  Accuracy:    %.4f (%.1f%%)\n', final_accuracy, final_accuracy*100);
fprintf('  Precision:   %.4f (%.1f%%)\n', final_precision, final_precision*100);
fprintf('  Recall:      %.4f (%.1f%%)\n', final_recall, final_recall*100);
fprintf('  F1 Score:    %.4f\n', final_f1);
fprintf('  Specificity: %.4f (%.1f%%)\n', final_specificity, final_specificity*100);
fprintf('  Confusion:   TP=%d, FP=%d, TN=%d, FN=%d\n', tp_final, fp_final, tn_final, fn_final);

%% 12. Plot Results - Single Figure with 3 Subplots (No Precision-Recall curve)
figure('Name', 'Normal vs Aneurysm Classification (Precision Optimized)', 'Position', [100, 100, 1400, 500]);

% Subplot 1: ROC Curve
subplot(1, 3, 1);
plot(X_roc, Y_roc, 'b-', 'LineWidth', 3);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1.5);
xlabel('False Positive Rate (1 - Specificity)', 'FontSize', 12);
ylabel('True Positive Rate (Sensitivity)', 'FontSize', 12);
title(sprintf('ROC Curve (AUC = %.3f)', overallAUC), 'FontSize', 14);
legend('Logistic Regression with SMOTE', 'Random Classifier', 'Location', 'southeast');
grid on;
axis square;

% Subplot 2: Feature Importance
subplot(1, 3, 2);

if isa(finalModel, 'ClassificationLinear')
    beta = finalModel.Beta;
    [~, idx] = sort(abs(beta), 'descend');
    topN = min(10, length(beta));
    
    for i = 1:topN
        if beta(idx(i)) > 0
            color = [1, 0.6, 0.6]; % Red for positive (aneurysm)
        else
            color = [0.6, 0.6, 1]; % Blue for negative (normal)
        end
        h = barh(i, beta(idx(i)));
        hold on;
        set(h, 'FaceColor', color, 'EdgeColor', 'k');
    end
    
    yticks(1:topN);
    yticklabels(featureNames(idx(1:topN)));
    xlabel('Coefficient Value', 'FontSize', 12);
    title('Top 10 Feature Importance', 'FontSize', 14);
    grid on;
    
    plot(NaN, NaN, 's', 'Color', [1, 0.6, 0.6], 'MarkerFaceColor', [1, 0.6, 0.6], 'MarkerSize', 10);
    plot(NaN, NaN, 's', 'Color', [0.6, 0.6, 1], 'MarkerFaceColor', [0.6, 0.6, 1], 'MarkerSize', 10);
    legend({'Positive (Aneurysm)', 'Negative (Normal)'}, 'Location', 'best', 'FontSize', 9);
end

% Subplot 3: Confusion Matrix
subplot(1, 3, 3);
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
title(sprintf('Confusion Matrix\nAccuracy: %.1f%% | Precision: %.1f%%', ...
    final_accuracy*100, final_precision*100), 'FontSize', 14);
axis square;

text(0.5, -0.3, sprintf('Recall: %.1f%% | F1: %.3f', ...
    final_recall*100, final_f1), ...
    'HorizontalAlignment', 'center', 'FontSize', 11, 'Units', 'normalized');

%% 13. Model Summary
fprintf('\n========== MODEL SUMMARY: PRECISION-OPTIMIZED with SMOTE ==========\n');
fprintf('Model Type: Logistic Regression with SMOTE + 5-Fold CV\n');
fprintf('SMOTE parameters: k=%d, target ratio=%.2f (minority = %.0f%% of majority)\n', ...
    k_neighbors_final, target_ratio_final, target_ratio_final*100);
fprintf('Class weights: Normal=%.2f, Aneurysm=%.2f\n', weightNormal_smote, weightAneurysm_smote);
fprintf('Regularization lambda: %.4f\n', lambda);
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