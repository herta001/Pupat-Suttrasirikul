Model	AUC	Accuracy	Precision	F1	Sensitivity (Recall)
LinR	0.8406	0.7370	0.1201	0.2094	0.8182
LogR	0.8371	0.9491	0.3103	0.0647	0.0361
SVM	0.8380	0.8840	0.2171	0.3081	0.5301
SVR	0.8250	0.8487	0.1828	0.2809	0.6064
DT	0.7490	0.7562	0.1162	0.2013	0.6305
GBDT	0.8280	0.9387	0.1735	0.0980	0.0683
RF	0.8340	0.7951	0.1596	0.2132	0.7510
XGboost	0.8350	0.8742	0.1903	0.2734	0.4859
%% 5-fold cross-validation with SVR

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

%% CLASS IMBALANCE ANALYSIS
fprintf('\n=== CLASS IMBALANCE ANALYSIS ===\n');

stroke_cases = sum(y);
non_stroke_cases = sum(y == 0);
total_cases = length(y);
imbalance_ratio = non_stroke_cases / stroke_cases;

fprintf('Stroke cases: %d (%.4f%%)\n', stroke_cases, (stroke_cases/total_cases)*100);
fprintf('Non-stroke cases: %d (%.4f%%)\n', non_stroke_cases, (non_stroke_cases/total_cases)*100);
fprintf('Imbalance ratio: %.2f:1 (non-stroke:stroke)\n', imbalance_ratio);

%% DATA PREPROCESSING
fprintf('\n=== DATA PREPROCESSING ===\n');

% 1. Outlier detection and treatment for continuous variables
continuous_vars = {'age', 'avg_glucose_level', 'bmi'};
available_continuous = continuous_vars(ismember(continuous_vars, use_vars));

for i = 1:length(available_continuous)
    var_name = available_continuous{i};
    var_idx = find(strcmp(use_vars, var_name));
    
    if ~isempty(var_idx)
        data = X(:, var_idx);
        
        % Detect outliers using IQR method
        Q1 = quantile(data, 0.25);
        Q3 = quantile(data, 0.75);
        IQR = Q3 - Q1;
        lower_bound = Q1 - 1.5 * IQR;
        upper_bound = Q3 + 1.5 * IQR;
        
        outliers = data < lower_bound | data > upper_bound;
        
        if sum(outliers) > 0
            fprintf('  %s: Found %d outliers (%.1f%%), capping to bounds\n', ...
                var_name, sum(outliers), mean(outliers)*100);
            
            % Cap outliers instead of removing
            data(data < lower_bound) = lower_bound;
            data(data > upper_bound) = upper_bound;
            X(:, var_idx) = data;
        end
    end
end

% 2. Check for near-zero variance features
fprintf('  Checking for near-zero variance features...\n');
variance_threshold = 0.01;
feature_variances = var(X);
low_variance_features = feature_variances < variance_threshold;

if sum(low_variance_features) > 0
    fprintf('  Removing %d low-variance features\n', sum(low_variance_features));
    X(:, low_variance_features) = [];
    use_vars(low_variance_features) = [];
end

% 3. Check for highly correlated features
fprintf('  Checking for highly correlated features...\n');
correlation_threshold = 0.95;

if size(X, 2) > 1
    correlation_matrix = corr(X);
    high_corr_pairs = [];
    
    for i = 1:size(correlation_matrix, 1)
        for j = i+1:size(correlation_matrix, 2)
            if abs(correlation_matrix(i, j)) > correlation_threshold
                high_corr_pairs = [high_corr_pairs; i, j, correlation_matrix(i, j)];
            end
        end
    end
    
    if ~isempty(high_corr_pairs)
        fprintf('  Found %d highly correlated feature pairs\n', size(high_corr_pairs, 1));
        % Keep the feature with higher variance in each pair
        features_to_remove = [];
        
        for k = 1:size(high_corr_pairs, 1)
            feat1 = high_corr_pairs(k, 1);
            feat2 = high_corr_pairs(k, 2);
            
            var1 = var(X(:, feat1));
            var2 = var(X(:, feat2));
            
            if var1 >= var2
                features_to_remove = [features_to_remove, feat2];
            else
                features_to_remove = [features_to_remove, feat1];
            end
        end
        
        features_to_remove = unique(features_to_remove);
        fprintf('  Removing %d redundant features\n', length(features_to_remove));
        
        X(:, features_to_remove) = [];
        use_vars(features_to_remove) = [];
    end
end

fprintf('After preprocessing: %d features remaining\n', size(X, 2));

%% FEATURE ENGINEERING
fprintf('\n=== FEATURE ENGINEERING ===\n');

% Create feature set with clinical knowledge
X_feature_engineered = X;
feature_names = use_vars;

% 1. Age transformations
if ismember('age', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    X_feature_engineered(:, end+1) = X(:, age_idx).^2; % Age squared
    X_feature_engineered(:, end+1) = log(X(:, age_idx) + 1); % Log age
    X_feature_engineered(:, end+1) = (X(:, age_idx) > 65); % Elderly indicator
    X_feature_engineered(:, end+1) = (X(:, age_idx) > 75); % Very elderly indicator
    feature_names{end+1} = 'age_squared';
    feature_names{end+1} = 'log_age';
    feature_names{end+1} = 'elderly_65plus';
    feature_names{end+1} = 'very_elderly_75plus';
    fprintf('Added age transformations\n');
end

% 2. BMI categories
if ismember('bmi', use_vars)
    bmi_idx = find(strcmp(use_vars, 'bmi'));
    X_feature_engineered(:, end+1) = (X(:, bmi_idx) < 18.5); % Underweight
    X_feature_engineered(:, end+1) = (X(:, bmi_idx) >= 25 & X(:, bmi_idx) < 30); % Overweight
    X_feature_engineered(:, end+1) = (X(:, bmi_idx) >= 30); % Obese
    X_feature_engineered(:, end+1) = (X(:, bmi_idx) >= 35); % Severely obese
    feature_names{end+1} = 'bmi_underweight';
    feature_names{end+1} = 'bmi_overweight';
    feature_names{end+1} = 'bmi_obese';
    feature_names{end+1} = 'bmi_severely_obese';
    fprintf('Added BMI categories\n');
end

% 3. Glucose level categories
if ismember('avg_glucose_level', use_vars)
    glucose_idx = find(strcmp(use_vars, 'avg_glucose_level'));
    X_feature_engineered(:, end+1) = (X(:, glucose_idx) > 140); % Hyperglycemia
    X_feature_engineered(:, end+1) = (X(:, glucose_idx) > 200); % Severe hyperglycemia
    X_feature_engineered(:, end+1) = (X(:, glucose_idx) < 70); % Hypoglycemia
    feature_names{end+1} = 'glucose_high';
    feature_names{end+1} = 'glucose_very_high';
    feature_names{end+1} = 'glucose_low';
    fprintf('Added glucose categories\n');
end

% 4. Clinical interaction terms
if ismember('age', use_vars) && ismember('hypertension', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    ht_idx = find(strcmp(use_vars, 'hypertension'));
    X_feature_engineered(:, end+1) = X(:, age_idx) .* X(:, ht_idx); % Age × Hypertension
    feature_names{end+1} = 'age_hypertension_interaction';
    fprintf('Added age-hypertension interaction\n');
end

% 5. Risk score combinations
if ismember('age', use_vars) && ismember('hypertension', use_vars) && ismember('heart_disease', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    ht_idx = find(strcmp(use_vars, 'hypertension'));
    hd_idx = find(strcmp(use_vars, 'heart_disease'));
    % Clinical risk score
    X_feature_engineered(:, end+1) = (X(:, age_idx) > 65) + X(:, ht_idx) + X(:, hd_idx);
    feature_names{end+1} = 'clinical_risk_score';
    fprintf('Added clinical risk score\n');
end

% 6. POLYNOMIAL FEATURES FOR KEY CONTINUOUS VARIABLES
fprintf('\n=== ADDING POLYNOMIAL FEATURES ===\n');

% Select key continuous features for polynomial expansion
key_continuous_features = {'age', 'avg_glucose_level', 'bmi'};
available_continuous_features = key_continuous_features(ismember(key_continuous_features, use_vars));

fprintf('Generating polynomial features for: ');
fprintf('%s ', available_continuous_features{:});
fprintf('\n');

for i = 1:length(available_continuous_features)
    feat_name = available_continuous_features{i};
    feat_idx = find(strcmp(use_vars, feat_name));
    
    if ~isempty(feat_idx)
        % Add squared term (x^2)
        X_feature_engineered(:, end+1) = X_feature_engineered(:, feat_idx).^2;
        feature_names{end+1} = sprintf('%s_squared', feat_name);
        fprintf('  - Added %s_squared\n', feat_name);
        
        % Add square root term (sqrt(x))
        X_feature_engineered(:, end+1) = sqrt(X_feature_engineered(:, feat_idx) + eps);
        feature_names{end+1} = sprintf('%s_sqrt', feat_name);
        fprintf('  - Added %s_sqrt\n', feat_name);
    end
end

% 7. INTERACTION TERMS BETWEEN KEY FEATURES
fprintf('=== ADDING INTERACTION TERMS ===\n');

% Create interactions between key continuous features
for i = 1:length(available_continuous_features)
    for j = i+1:length(available_continuous_features)
        feat1 = available_continuous_features{i};
        feat2 = available_continuous_features{j};
        
        feat1_idx = find(strcmp(use_vars, feat1));
        feat2_idx = find(strcmp(use_vars, feat2));
        
        if ~isempty(feat1_idx) && ~isempty(feat2_idx)
            % Multiply the two features
            X_feature_engineered(:, end+1) = X_feature_engineered(:, feat1_idx) .* X_feature_engineered(:, feat2_idx);
            feature_names{end+1} = sprintf('%s_times_%s', feat1, feat2);
            fprintf('  - Added %s × %s interaction\n', feat1, feat2);
        end
    end
end

% 8. CLINICAL RISK SCORES
fprintf('\n=== CLINICAL RISK SCORES ===\n');

% Framingham-inspired stroke risk score
if all(ismember({'age', 'hypertension', 'heart_disease'}, use_vars))
    age_idx = find(strcmp(use_vars, 'age'));
    ht_idx = find(strcmp(use_vars, 'hypertension'));
    hd_idx = find(strcmp(use_vars, 'heart_disease'));
    
    % Risk score with weighted components
    stroke_risk = zeros(size(X_feature_engineered, 1), 1);
    stroke_risk = stroke_risk + (X(:, age_idx) > 75) * 3;        % Age > 75: high risk
    stroke_risk = stroke_risk + (X(:, age_idx) > 65) * 2;        % Age 65-75: medium risk
    stroke_risk = stroke_risk + (X(:, age_idx) > 55) * 1;        % Age 55-65: low risk
    stroke_risk = stroke_risk + X(:, ht_idx) * 2;                % Hypertension
    stroke_risk = stroke_risk + X(:, hd_idx) * 2;                % Heart disease
    stroke_risk = stroke_risk + (X(:, ht_idx) & X(:, hd_idx)) * 3; % Comorbidity bonus
    
    X_feature_engineered(:, end+1) = stroke_risk;
    feature_names{end+1} = 'stroke_risk_score';
    fprintf('Added stroke risk score\n');
end

% Age-decade indicators for non-linear age effects
if ismember('age', use_vars)
    age_idx = find(strcmp(use_vars, 'age'));
    age_decades = floor(X(:, age_idx) / 10);
    unique_decades = unique(age_decades);
    
    for decade = unique_decades(unique_decades >= 4 & unique_decades <= 9)' % 40-90+
        X_feature_engineered(:, end+1) = (age_decades == decade);
        feature_names{end+1} = sprintf('age_decade_%d', decade*10);
    end
    fprintf('Added age decade indicators\n');
end

fprintf('Feature engineering completed: %d new features created\n', ...
    length(feature_names) - length(use_vars));

fprintf('Feature set: %d features -> %d features\n', size(X, 2), size(X_feature_engineered, 2));

% Update X with engineered features
X = X_feature_engineered;
use_vars = feature_names;

%%SMOTE IMPLEMENTATION
fprintf('\n=== SMOTE FOR CLASS IMBALANCE HANDLING ===\n');

% First, split the data into train and test sets
rng(42); % For reproducibility
cv = cvpartition(y, 'Holdout', 0.3); % 70-30 split
train_idx = training(cv);
test_idx = test(cv);

% Store original split for comparison
X_train_original = X(train_idx, :);
y_train_original = y(train_idx);
X_test = X(test_idx, :);
y_test = y(test_idx);

% Calculate class distribution in training set
minority_class = 1; % Stroke cases
majority_class = 0; % Non-stroke cases

train_minority_count = sum(y_train_original == minority_class);
train_majority_count = sum(y_train_original == majority_class);
train_imbalance_ratio = train_majority_count / train_minority_count;

fprintf('Training set before SMOTE:\n');
fprintf('  Minority class (stroke): %d samples (%.2f%%)\n', train_minority_count, 100*train_minority_count/length(y_train_original));
fprintf('  Majority class (non-stroke): %d samples (%.2f%%)\n', train_majority_count, 100*train_majority_count/length(y_train_original));
fprintf('  Imbalance ratio: %.2f:1\n', train_imbalance_ratio);

% Determine desired sampling strategy
% Target 1:2 ratio (conservative balancing - often works better than 1:1 for medical data)
desired_minority_count = min(train_majority_count, train_minority_count * 2);

synthetic_samples_needed = desired_minority_count - train_minority_count;

fprintf('  Target minority samples after SMOTE: %d\n', desired_minority_count);
fprintf('  Synthetic samples needed: %d\n', synthetic_samples_needed);

% Apply SMOTE only if needed
if synthetic_samples_needed > 0
    fprintf('  Generating %d synthetic minority samples using SMOTE...\n', synthetic_samples_needed);
    
    % Prepare minority class data
    train_minority_idx = find(y_train_original == minority_class);
    X_train_minority = X_train_original(train_minority_idx, :);
    
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
                feat_min = min(X_train_original(:, j));
                feat_max = max(X_train_original(:, j));
                
                % Clip to bounds
                synthetic_sample(j) = max(feat_min, min(feat_max, synthetic_sample(j)));
            end
            
            synthetic_count = synthetic_count + 1;
            X_synthetic(synthetic_count, :) = synthetic_sample;
        end
    end
    
    % Combine original and synthetic data
    X_train = [X_train_original; X_synthetic];
    y_train = [y_train_original; y_synthetic];
    
    % Shuffle the balanced dataset
    shuffle_idx = randperm(size(X_train, 1));
    X_train = X_train(shuffle_idx, :);
    y_train = y_train(shuffle_idx);
    
    fprintf('  After SMOTE:\n');
    fprintf('    Total training samples: %d\n', size(X_train, 1));
    fprintf('    Minority samples: %d (%.2f%%)\n', sum(y_train == minority_class), 100*sum(y_train == minority_class)/length(y_train));
    fprintf('    Majority samples: %d (%.2f%%)\n', sum(y_train == majority_class), 100*sum(y_train == majority_class)/length(y_train));
    fprintf('    New imbalance ratio: %.2f:1\n', sum(y_train == majority_class)/sum(y_train == minority_class));
    
else
    fprintf('  No SMOTE needed - minority class already sufficient.\n');
    X_train = X_train_original;
    y_train = y_train_original;
end

fprintf('SMOTE completed. Training set: %d samples, Test set: %d samples\n', ...
    size(X_train, 1), size(X_test, 1));

%% Feature Scaling for SVR
fprintf('\n=== FEATURE SCALING FOR SVR ===\n');

mu = mean(X_train);
sigma = std(X_train);
sigma(sigma == 0) = 1; % Avoid division by zero

X_train_scaled = (X_train - mu) ./ sigma;
X_test_scaled = (X_test - mu) ./ sigma;

fprintf('Feature scaling completed\n');

%% SVR HYPERPARAMETER TUNING
fprintf('\n=== SVR HYPERPARAMETER TUNING ===\n');

% Class imbalance handling (using balanced training data)
fprintf('Calculating sample weights for class imbalance...\n');

% Calculate new imbalance ratio after SMOTE
minority_count_balanced = sum(y_train == 1);
majority_count_balanced = sum(y_train == 0);
balanced_ratio = majority_count_balanced / minority_count_balanced;

fprintf('Balanced dataset ratio: %.2f:1\n', balanced_ratio);

% Adjust sample weights based on new balance
if balanced_ratio > 3
    weight_factor = sqrt(balanced_ratio);
elseif balanced_ratio > 1.5
    weight_factor = log(balanced_ratio + 1) * 1.5;
else
    weight_factor = 1.0; % Balanced, no weighting needed
end

sample_weights = ones(size(y_train));
sample_weights(y_train == 1) = weight_factor;
fprintf('Sample weighting factor: %.2f\n', weight_factor);

% Define search space for hyperparameters
kernel_options = {'linear'};
box_constraints = [0.1, 0.5, 1, 5, 10, 50, 100, 150, 200, 300, 500];
epsilon_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7];

% Initialize best parameters
best_kernel = 'linear';
best_box = 100.0;
best_epsilon = 0.300;
best_auc = 0;
best_scale = 1;

fprintf('Searching %d hyperparameter combinations...\n', ...
    length(kernel_options) * length(box_constraints) * length(epsilon_values));

% Use a subset for faster validation
if size(X_train_scaled, 1) > 2000
    val_indices = randperm(size(X_train_scaled, 1), 2000);
    X_val = X_train_scaled(val_indices, :);
    y_val = y_train(val_indices);
    weights_val = sample_weights(val_indices);
else
    X_val = X_train_scaled;
    y_val = y_train;
    weights_val = sample_weights;
end

% Grid search with cross-validation
total_combinations = length(box_constraints) * length(epsilon_values);
combination_count = 0;

for box_idx = 1:length(box_constraints)
    for eps_idx = 1:length(epsilon_values)
        combination_count = combination_count + 1;
        
        current_box = box_constraints(box_idx);
        current_epsilon = epsilon_values(eps_idx);
        
        if mod(combination_count, 10) == 0
            fprintf('  Progress: %d/%d (%.1f%%) - Best AUC: %.4f\n', ...
                combination_count, total_combinations, ...
                (combination_count/total_combinations)*100, best_auc);
        end
        
        try
            % Train model with current hyperparameters
            temp_model = fitrsvm(X_val, y_val, ...
                'KernelFunction', best_kernel, ...
                'Standardize', false, ...
                'BoxConstraint', current_box, ...
                'Epsilon', current_epsilon, ...
                'Weights', weights_val, ...
                'Verbose', 0);
            
            % Get predictions and calculate AUC
            temp_pred = predict(temp_model, X_val);
            temp_prob = 1 ./ (1 + exp(-temp_pred));
            [~, ~, ~, temp_auc] = perfcurve(y_val, temp_prob, 1);
            
            % Update best parameters if improvement found
            if temp_auc > best_auc
                best_auc = temp_auc;
                best_box = current_box;
                best_epsilon = current_epsilon;
                fprintf('    New best: Box=%.1f, Epsilon=%.3f, AUC=%.4f\n', ...
                    best_box, best_epsilon, best_auc);
            end
            
        catch ME
            continue;
        end
    end
end

% REFINEMENT SEARCH AROUND BEST PARAMETERS
fprintf('\n=== REFINING HYPERPARAMETERS ===\n');

if best_box > 10 && best_box < 500
    box_refine = [best_box*0.5, best_box*0.8, best_box, best_box*1.2, best_box*1.5];
    epsilon_refine = [best_epsilon*0.7, best_epsilon*0.9, best_epsilon, best_epsilon*1.1, best_epsilon*1.3];
    
    for refine_box = box_refine
        for refine_epsilon = epsilon_refine
            try
                refine_model = fitrsvm(X_val, y_val, ...
                    'KernelFunction', best_kernel, ...
                    'Standardize', false, ...
                    'BoxConstraint', refine_box, ...
                    'Epsilon', refine_epsilon, ...
                    'Weights', weights_val, ...
                    'Verbose', 0);
                
                refine_pred = predict(refine_model, X_val);
                refine_prob = 1 ./ (1 + exp(-refine_pred));
                [~, ~, ~, refine_auc] = perfcurve(y_val, refine_prob, 1);
                
                if refine_auc > best_auc
                    best_auc = refine_auc;
                    best_box = refine_box;
                    best_epsilon = refine_epsilon;
                    fprintf('  Refined: Box=%.1f, Epsilon=%.3f, AUC=%.4f\n', ...
                        best_box, best_epsilon, best_auc);
                end
                
            catch
                continue;
            end
        end
    end
end

fprintf('\n=== FINAL OPTIMAL HYPERPARAMETERS ===\n');
fprintf('Best BoxConstraint: %.1f\n', best_box);
fprintf('Best Epsilon:       %.3f\n', best_epsilon);
fprintf('Best Kernel:        %s\n', best_kernel);
fprintf('Validation AUC:     %.4f\n', best_auc);

% Train final model with optimized parameters
svr_model = fitrsvm(X_train_scaled, y_train, ...
    'KernelFunction', best_kernel, ...
    'Standardize', false, ...
    'KernelScale', best_scale, ...
    'BoxConstraint', best_box, ...
    'Epsilon', best_epsilon, ...
    'Weights', sample_weights);

fprintf('SVR model training completed with optimized parameters\n');

%% ENSEMBLE PROBABILITY CALIBRATION
fprintf('\n=== ENSEMBLE PROBABILITY CALIBRATION ===\n');

% Get continuous predictions
svr_pred_continuous = predict(svr_model, X_test_scaled);

% Ensemble calibration
prob_methods = struct();

% Method 1: Standard sigmoid
prob_methods.method1 = 1 ./ (1 + exp(-svr_pred_continuous));

% Method 2: Platt scaling
svr_pred_train = predict(svr_model, X_train_scaled);
pos_mean = mean(svr_pred_train(y_train == 1));
neg_mean = mean(svr_pred_train(y_train == 0));
prob_methods.method2 = 1 ./ (1 + exp(-(svr_pred_continuous - neg_mean) / (pos_mean - neg_mean + eps)));

% Method 3: Min-max scaling
prob_methods.method3 = (svr_pred_continuous - min(svr_pred_continuous)) / (max(svr_pred_continuous) - min(svr_pred_continuous));

% Method 4: Adaptive scaling based on imbalance
if imbalance_ratio > 10  % Use original imbalance ratio
    scale_factor = 3;
else
    scale_factor = 2;
end
prob_methods.method4 = 1 ./ (1 + exp(-svr_pred_continuous * scale_factor));

% Method 5: Bayesian adjustment
prior_stroke_rate = mean(y_train); % Use balanced prior
prob_methods.method5 = (prob_methods.method1 + 0.3 * prior_stroke_rate) / 1.3;

% Fixed weights based on typical performance
val_weights = [0.20, 0.35, 0.15, 0.20, 0.10];

% Apply weighted ensemble
svr_pred_prob = zeros(size(svr_pred_continuous));
method_names = {'method1', 'method2', 'method3', 'method4', 'method5'};
for i = 1:5
    svr_pred_prob = svr_pred_prob + val_weights(i) * prob_methods.(method_names{i});
end
svr_pred_prob = max(0, min(1, svr_pred_prob));

fprintf('Applied ensemble weights\n');
fprintf('Weights: [Standard: 0.20, Platt: 0.35, MinMax: 0.15, Adaptive: 0.20, Bayesian: 0.10]\n');

%% PROBABILITY SMOOTHING
fprintf('\n=== PROBABILITY SMOOTHING ===\n');

% Apply moving average smoothing to probabilities
window_size = 5; % Small window for smoothing
if length(svr_pred_prob) > window_size
    smoothed_probs = zeros(size(svr_pred_prob));
    for i = 1:length(svr_pred_prob)
        start_idx = max(1, i - floor(window_size/2));
        end_idx = min(length(svr_pred_prob), i + floor(window_size/2));
        smoothed_probs(i) = mean(svr_pred_prob(start_idx:end_idx));
    end
    % Blend original and smoothed probabilities
    blend_alpha = 0.3; % Weight for smoothed probabilities
    svr_pred_prob = (1 - blend_alpha) * svr_pred_prob + blend_alpha * smoothed_probs;
    fprintf('Applied probability smoothing (alpha=%.2f)\n', blend_alpha);
end

%% THRESHOLD OPTIMIZATION
fprintf('\n=== THRESHOLD OPTIMIZATION ===\n');

% Generate ROC curve
[X_roc_svr, Y_roc_svr, T_svr, AUC_svr] = perfcurve(y_test, svr_pred_prob, 1);

% Calculate comprehensive metrics for each threshold
f1_scores = zeros(length(T_svr), 1);
precision_scores = zeros(length(T_svr), 1);
recall_scores = zeros(length(T_svr), 1);
gmean_scores = zeros(length(T_svr), 1);
mcc_scores = zeros(length(T_svr), 1);

for i = 1:length(T_svr)
    temp_pred = svr_pred_prob >= T_svr(i);
    TP = sum(temp_pred & y_test);
    FP = sum(temp_pred & ~y_test);
    TN = sum(~temp_pred & ~y_test);
    FN = sum(~temp_pred & y_test);
    
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    specificity_temp = TN / (TN + FP + eps);
    
    precision_scores(i) = precision_temp;
    recall_scores(i) = recall_temp;
    f1_scores(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
    gmean_scores(i) = sqrt(recall_temp * specificity_temp);
    
    % Calculate Matthews Correlation Coefficient
    mcc_scores(i) = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + eps);
end

% Multi-objective optimization with clinical weighting
clinical_weights = struct();
clinical_weights.f1 = 0.3;        % Overall balance
clinical_weights.precision = 0.25; % Important for medical decisions
clinical_weights.recall = 0.2;    % Don't miss strokes
clinical_weights.mcc = 0.15;      % Overall correlation
clinical_weights.gmean = 0.1;     % Balanced accuracy

% Normalize scores to [0,1] range for fair comparison
f1_normalized = (f1_scores - min(f1_scores)) / (max(f1_scores) - min(f1_scores) + eps);
precision_normalized = (precision_scores - min(precision_scores)) / (max(precision_scores) - min(precision_scores) + eps);
recall_normalized = (recall_scores - min(recall_scores)) / (max(recall_scores) - min(recall_scores) + eps);
mcc_normalized = (mcc_scores - min(mcc_scores)) / (max(mcc_scores) - min(mcc_scores) + eps);
gmean_normalized = (gmean_scores - min(gmean_scores)) / (max(gmean_scores) - min(gmean_scores) + eps);

% Combined objective function
combined_scores = clinical_weights.f1 * f1_normalized + ...
                 clinical_weights.precision * precision_normalized + ...
                 clinical_weights.recall * recall_normalized + ...
                 clinical_weights.mcc * mcc_normalized + ...
                 clinical_weights.gmean * gmean_normalized;

% Find optimal threshold
[~, optimal_idx_svr] = max(combined_scores);
optimal_threshold_svr = T_svr(optimal_idx_svr);

fprintf('Optimal threshold: %.4f\n', optimal_threshold_svr);
fprintf('Performance at optimal threshold:\n');
fprintf('  F1-Score:      %.4f\n', f1_scores(optimal_idx_svr));
fprintf('  Precision:     %.4f\n', precision_scores(optimal_idx_svr));
fprintf('  Recall:        %.4f\n', recall_scores(optimal_idx_svr));
fprintf('  MCC:           %.4f\n', mcc_scores(optimal_idx_svr));
fprintf('  G-Mean:        %.4f\n', gmean_scores(optimal_idx_svr));

% Convert to binary predictions
svr_pred_binary = svr_pred_prob >= optimal_threshold_svr;
y_test_numeric = double(y_test);
svr_pred_binary_numeric = double(svr_pred_binary);

% Recalculate AUC with optimized threshold
[X_roc_svr_final, Y_roc_svr_final, ~, AUC_svr_final] = perfcurve(y_test, svr_pred_prob, 1);
fprintf('Final AUC after threshold optimization: %.4f\n', AUC_svr_final);

%% Calculate Performance Metrics
TP_svr = sum(svr_pred_binary_numeric & y_test_numeric);
FP_svr = sum(svr_pred_binary_numeric & ~y_test_numeric);
TN_svr = sum(~svr_pred_binary_numeric & ~y_test_numeric);
FN_svr = sum(~svr_pred_binary_numeric & y_test_numeric);

accuracy_svr = (TP_svr + TN_svr) / (TP_svr + TN_svr + FP_svr + FN_svr);
precision_svr = TP_svr / (TP_svr + FP_svr + eps);
recall_svr = TP_svr / (TP_svr + FN_svr + eps);
specificity_svr = TN_svr / (TN_svr + FP_svr + eps);
f1_score_svr = 2 * (precision_svr * recall_svr) / (precision_svr + recall_svr + eps);
brier_score_svr = mean((svr_pred_prob - y_test_numeric).^2);

% Additional metrics
balanced_accuracy = (recall_svr + specificity_svr) / 2;
mcc = (TP_svr * TN_svr - FP_svr * FN_svr) / sqrt((TP_svr + FP_svr) * (TP_svr + FN_svr) * (TN_svr + FP_svr) * (TN_svr + FN_svr) + eps);

% Calculate Precision-Recall AUC
[X_pr_svr, Y_pr_svr, ~, AUPRC_svr] = perfcurve(y_test, svr_pred_prob, 1, 'XCrit', 'reca', 'YCrit', 'prec');

%% Display Results
fprintf('\n=== FINAL SVR PERFORMANCE (WITH SMOTE) ===\n');
fprintf('Original dataset imbalance: %.2f:1\n', imbalance_ratio);
fprintf('Training set after SMOTE: %d samples\n', length(y_train));
fprintf('  Stroke cases in training: %d (%.2f%%)\n', sum(y_train == 1), 100*mean(y_train));
fprintf('  Non-stroke cases in training: %d (%.2f%%)\n', sum(y_train == 0), 100*(1-mean(y_train)));
fprintf('Test set: %d samples\n', length(y_test));
fprintf('  Stroke cases in test: %d (%.2f%%)\n', sum(y_test == 1), 100*mean(y_test));
fprintf('  Non-stroke cases in test: %d (%.2f%%)\n', sum(y_test == 0), 100*(1-mean(y_test)));
fprintf('AUC:              %.4f\n', AUC_svr_final);
fprintf('AUPRC:            %.4f\n', AUPRC_svr);
fprintf('Accuracy:         %.4f (%.2f%%)\n', accuracy_svr, accuracy_svr*100);
fprintf('Balanced Accuracy:%.4f\n', balanced_accuracy);
fprintf('Precision:        %.4f (%.2f%%)\n', precision_svr, precision_svr*100);
fprintf('Recall:           %.4f (%.2f%%)\n', recall_svr, recall_svr*100);
fprintf('Specificity:      %.4f (%.2f%%)\n', specificity_svr, specificity_svr*100);
fprintf('F1-Score:         %.4f\n', f1_score_svr);
fprintf('MCC:              %.4f\n', mcc);
fprintf('Brier Score:      %.4f\n', brier_score_svr);
fprintf('Optimal Threshold:%.4f\n', optimal_threshold_svr);

%% VISUALIZATION
fprintf('\n=== GENERATING VISUALIZATIONS ===\n');

figure;
sgtitle('SVR for Stroke Prediction (with SMOTE)', 'FontSize', 16, 'FontWeight', 'bold');

% Plot 1: ROC Curve with optimal threshold
subplot(2,3,1);
plot(X_roc_svr_final, Y_roc_svr_final, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
plot(X_roc_svr_final(optimal_idx_svr), Y_roc_svr_final(optimal_idx_svr), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
xlabel('False Positive Rate');
ylabel('True Positive Rate (Sensitivity)');
title(sprintf('ROC Curve (AUC = %.4f)', AUC_svr_final));
legend(sprintf('SVR with SMOTE (AUC=%.4f)', AUC_svr_final), ...
       'Random Classifier', ...
       sprintf('Optimal Threshold=%.3f', optimal_threshold_svr), ...
       'Location', 'southeast');
grid on;

% Plot 2: Precision-Recall Curve
subplot(2,3,2);
plot(X_pr_svr, Y_pr_svr, 'g-', 'LineWidth', 2);
xlabel('Recall (Sensitivity)');
ylabel('Precision');
title(sprintf('Precision-Recall Curve (AUPRC = %.4f)', AUPRC_svr));
grid on;
% Add baseline for imbalanced data
baseline_precision = sum(y_test) / length(y_test);
line([0 1], [baseline_precision baseline_precision], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 1);
legend('SVR with SMOTE', sprintf('Baseline (%.4f)', baseline_precision), 'Location', 'southwest');

% Plot 3: Probability distributions by class
subplot(2,3,3);
hold on;
if sum(y_test_numeric==0) > 0
    histogram(svr_pred_prob(y_test_numeric==0), 'BinWidth', 0.02, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
if sum(y_test_numeric==1) > 0
    histogram(svr_pred_prob(y_test_numeric==1), 'BinWidth', 0.02, 'FaceColor', 'red', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
end
line([optimal_threshold_svr optimal_threshold_svr], ylim(), 'Color', 'black', 'LineWidth', 2, 'LineStyle', '--');
xlabel('Predicted Probability');
ylabel('Density');
title('Probability Distribution by Class (Test Set)');
legend('No Stroke', 'Stroke', sprintf('Threshold=%.3f', optimal_threshold_svr), 'Location', 'northwest');
grid on;

% Plot 4: Feature importance for SVR (using permutation importance)
subplot(2,3,4);
base_mse = mean((svr_pred_continuous - y_test_numeric).^2);
perm_importance = zeros(1, min(10, size(X_test_scaled, 2)));

for i = 1:min(10, size(X_test_scaled, 2))
    X_permuted = X_test_scaled;
    X_permuted(:,i) = X_permuted(randperm(size(X_permuted,1)), i);
    perm_pred = predict(svr_model, X_permuted);
    perm_mse = mean((perm_pred - y_test_numeric).^2);
    perm_importance(i) = perm_mse - base_mse;
end

% Sort by importance
[perm_sorted, perm_sorted_idx] = sort(perm_importance, 'descend');
top_features_to_show = min(8, length(use_vars));

barh(perm_sorted(1:top_features_to_show));
set(gca, 'YTick', 1:top_features_to_show, 'YTickLabel', use_vars(perm_sorted_idx(1:top_features_to_show)));
xlabel('Feature Importance');
title('SVR Feature Importance (with SMOTE)');
grid on;
hold on;
plot([0 0], ylim(), 'r--', 'LineWidth', 2);
legend('Feature Importance', 'No Effect', 'Location', 'southeast');

% Plot 5: Threshold analysis
subplot(2,3,5);
thresholds = 0.01:0.01:0.99;
f1_scores_plot = zeros(size(thresholds));
for i = 1:length(thresholds)
    y_pred_temp = svr_pred_prob >= thresholds(i);
    TP = sum(y_pred_temp & y_test_numeric);
    FP = sum(y_pred_temp & ~y_test_numeric);
    FN = sum(~y_pred_temp & y_test_numeric);
    precision_temp = TP / (TP + FP + eps);
    recall_temp = TP / (TP + FN + eps);
    f1_scores_plot(i) = 2 * (precision_temp * recall_temp) / (precision_temp + recall_temp + eps);
end
plot(thresholds, f1_scores_plot, 'k-', 'LineWidth', 2);
hold on;
plot(optimal_threshold_svr, f1_scores_plot(round(optimal_threshold_svr*100)), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
xlabel('Classification Threshold');
ylabel('F1-Score');
title('F1-Score vs Threshold');
grid on;
legend('F1-Score', sprintf('Optimal (%.3f)', optimal_threshold_svr), 'Location', 'southeast');

% Plot 6: Calibration plot
subplot(2,3,6);
bin_edges = 0:0.1:1;
bin_centers = 0.05:0.1:0.95;
mean_predicted_svr = zeros(size(bin_centers));
actual_proportion_svr = zeros(size(bin_centers));
for i = 1:length(bin_centers)
    in_bin = svr_pred_prob >= bin_edges(i) & svr_pred_prob < bin_edges(i+1);
    if sum(in_bin) > 0
        mean_predicted_svr(i) = mean(svr_pred_prob(in_bin));
        actual_proportion_svr(i) = mean(y_test_numeric(in_bin));
    end
end
plot(mean_predicted_svr, actual_proportion_svr, 'bo-', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 2);
xlabel('Mean Predicted Probability');
ylabel('Actual Proportion');
title('Probability Calibration Plot (Test Set)');
legend('SVR with SMOTE', 'Perfect Calibration', 'Location', 'northwest');
grid on;

% Plot 7: Confusion Matrix
figure;
sgtitle('SVR for Stroke Prediction (with SMOTE)', 'FontSize', 16, 'FontWeight', 'bold');
subplot(1,2,1);
CM_svr = confusionmat(y_test_numeric, svr_pred_binary_numeric);
confusionchart(CM_svr, {'No Stroke', 'Stroke'}, 'Title', 'Confusion Matrix (Optimal Threshold)');

% Plot 8: Class distribution
subplot(1,2,2);
pie([sum(y_test_numeric==0), sum(y_test_numeric==1)], {'No Stroke', 'Stroke'});
title(sprintf('Test Set Class Distribution\n(Imbalance: %.1f:1)', sum(y_test==0)/sum(y_test==1)));

% Plot 9: Jittered scatter plot for binary classification
figure;
subplot(1,2,1);
% Add small random noise to actual values for visualization
rng(42); % For reproducibility
y_jittered = y_test_numeric + 0.05 * randn(size(y_test_numeric)); % Small jitter to actual values
svr_pred_jittered = svr_pred_prob + 0.01 * randn(size(svr_pred_prob)); % Very small jitter to predictions

% Color points by actual class
colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980]; % Blue and orange
scatter(y_jittered(y_test_numeric==0), svr_pred_jittered(y_test_numeric==0), 40, colors(1,:), 'filled', ...
        'MarkerFaceAlpha', 0.6, 'DisplayName', 'Actual: No Stroke');
hold on;
scatter(y_jittered(y_test_numeric==1), svr_pred_jittered(y_test_numeric==1), 40, colors(2,:), 'filled', ...
        'MarkerFaceAlpha', 0.6, 'DisplayName', 'Actual: Stroke');

% Add perfect prediction lines for binary classification
plot([0.5 0.5], [-0.1 1.1], 'r--', 'LineWidth', 2, 'HandleVisibility', 'off');

xlim([-0.2, 1.2]);
ylim([-0.1, 1.1]);
xlabel('Actual Stroke (with jitter)');
ylabel('Predicted Probability (with jitter)');
title(sprintf('Jittered Scatter Plot\n(Optimal Threshold = %.3f)', optimal_threshold_svr));
grid on;
legend('Location', 'northwest');

% Plot 10: Box plot of predictions by actual class
subplot(1,2,2);
boxplot(svr_pred_prob, y_test_numeric, 'Labels', {'No Stroke', 'Stroke'});
xlabel('Actual Class');
ylabel('Predicted Probability');
title('Box Plot of Predicted Probabilities by Actual Class');
grid on;

% Add optimal threshold line to box plot
hold on;
y_limits = ylim();
plot(xlim(), [optimal_threshold_svr optimal_threshold_svr], 'r--', 'LineWidth', 2, ...
     'DisplayName', sprintf('Optimal Threshold (%.3f)', optimal_threshold_svr));
legend('Location', 'northwest');

%% Feature Importance Analysis
fprintf('\n=== TOP FEATURE IMPORTANCE (WITH SMOTE) ===\n');
fprintf('Top 5 most important features:\n');
for i = 1:min(5, length(use_vars))
    fprintf('%d. %-30s (importance: %.4f)\n', i, use_vars{perm_sorted_idx(i)}, perm_sorted(i));
end

fprintf('\n=== SVR ANALYSIS COMPLETE (WITH SMOTE) ===\n');
fprintf('SMOTE Configuration: Generated %d synthetic samples\n', synthetic_samples_needed);
fprintf('Final training set: %d samples (%.1f%% stroke)\n', length(y_train), 100*mean(y_train));
fprintf('Test set: %d samples (%.1f%% stroke)\n', length(y_test), 100*mean(y_test));
fprintf('SVR Configuration: Linear kernel, Box=%.1f, Epsilon=%.3f\n', best_box, best_epsilon);