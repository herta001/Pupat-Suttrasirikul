%% 
clc; clear all; % For clearing workspace

% Load clinical datasets
Data2 = readtable('healthcare-dataset-stroke-data.csv');

% Display data sizes
fprintf('=== DATA SIZES ===\n');
fprintf('healthcare-dataset-stroke-data.csv: %d rows × %d columns\n', size(Data2, 1), size(Data2, 2));
fprintf('==========================\n');

% Optional: Display total statistics
total_rows = size(Data2, 1);
total_columns = size(Data2, 2);
fprintf('Dimension of dataset: %d rows, %d columns\n', total_rows, total_columns);

% ===== USING FULL STROKE DATASET =====
fprintf('\n=== USING FULL STROKE DATASET (Data2) ===\n');

% Display dataset info
fprintf('Dataset size: %d rows × %d columns\n', size(Data2, 1), size(Data2, 2));
fprintf('Using complete dataset for analysis\n');

% Display first few rows
fprintf('\nFirst 5 rows of the dataset:\n');
disp(Data2(1:min(5, size(Data2, 1)), :));

fprintf('\nFull dataset ready for analysis...\n');
