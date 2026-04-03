% DWI BRAIN ARTERY FEATURE EXTRACTION - 12 REGION ANALYSIS

clear all; clc; close all;

% IMAGE LOADING AND BATCH PROCESSING
% Load ALL DWI images from current directory
files = dir('*.jpg');  % Change extension as needed (png, tif, dcm)
n_slices = length(files);
fprintf('Found %d DWI image slices to process\n', n_slices);

if n_slices == 0
    error('No DWI images found in current directory');
end

% Initialize master table as empty
all_features_table = table();

% Create progress bar
h_waitbar = waitbar(0, 'Processing DWI slices...');

% Define patient status - NORMAL (no aneurysm)
patient_status = 'no';
patient_group = 'Normal_DWI';

% DWI-SPECIFIC PARAMETERS
% Using 12 regions (3x4 grid) with overlap for smooth blending
target_regions = 12;
region_overlap = 12;  % Pixels overlap
edge_erosion = 4;     % Less erosion for DWI (keep more brain)
min_vessel_size = 10; % Minimum artery size

fprintf('\n%s\n', repmat('-', 1, 80));
fprintf('DWI ARTERY FEATURE EXTRACTION - 12 REGION ANALYSIS\n');
fprintf('%s\n', repmat('-', 1, 80));

fprintf('\nDWI-SPECIFIC CONFIGURATION:\n');
fprintf('  Regions per slice: %d (3x4 grid with %d-pixel overlap)\n', target_regions, region_overlap);
fprintf('  Edge suppression: %d-pixel erosion\n', edge_erosion);
fprintf('  Minimum artery size: %d pixels\n', min_vessel_size);
fprintf('  Total regions to process: %d\n', target_regions * n_slices);

% Process each slice
for slice_idx = 1:n_slices
    % Update progress
    waitbar(slice_idx/n_slices, h_waitbar, sprintf('Processing DWI slice %d/%d: %s', ...
        slice_idx, n_slices, files(slice_idx).name));
    
    fprintf('\n%s\n', repmat('-', 1, 80));
    fprintf('PROCESSING DWI SLICE %d/%d: %s\n', slice_idx, n_slices, files(slice_idx).name);
    fprintf('%s\n', repmat('-', 1, 80));
    
    % Load and preprocess current slice
    img = imread(files(slice_idx).name);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    original = im2double(img);

    % DWI PREPROCESSING PIPELINE (ADAPTED FOR DWI)
    fprintf('\nStep 1: DWI preprocessing...\n');
    
    % DWI-specific wavelet denoising (higher threshold for DWI noise)
    brain_denoised = wavelet_denoise_dwi(original, 2.2);
    
    % DWI-specific skull stripping (adjusted thresholds for DWI)
    brain_mask = skull_strip_dwi_improved(brain_denoised);
    
    % Create brain interior mask (remove edge artifacts)
    interior_mask = create_brain_interior_mask_dwi(brain_mask, edge_erosion);
    
    % Apply mask to get brain only
    brain_only = brain_denoised .* brain_mask;
    
    % DWI-specific CLAHE enhancement (more conservative)
    brain_enhanced = adapthisteq(brain_only, ...
        'NumTiles', [8 8], ...
        'ClipLimit', 0.015, ...  % Lower clip limit for DWI
        'Distribution', 'rayleigh', ...
        'Alpha', 0.4);
    brain_enhanced(~brain_mask) = 0;

    % DWI ARTERY SEGMENTATION (12-REGION ANALYSIS)
    fprintf('\nStep 2: Performing DWI artery segmentation with %d regions...\n', target_regions);
    
    % Use DWI-specific artery extraction
    [artery_mask, artery_map] = extract_dwi_arteries_12regions(brain_enhanced, brain_mask, interior_mask, target_regions);

    % DWI REGION-BASED FEATURE EXTRACTION - 12 REGIONS
    fprintf('\nStep 3: Extracting artery features from %d regions...\n', target_regions);
    
    % Extract features from each region using DWI-specific function
    region_features = extractArteryFeaturesDWI_12regions(artery_mask, interior_mask, brain_enhanced, ...
        slice_idx, files(slice_idx).name, patient_status, patient_group, min_vessel_size);
    
    % Add to master table
    if isempty(all_features_table)
        all_features_table = region_features;
    else
        % Ensure consistent variable types before concatenation
        region_features = ensureConsistentTypes(region_features, all_features_table);
        all_features_table = [all_features_table; region_features];
    end
    
    fprintf('\nSlice %d complete - Generated %d region samples\n', ...
        slice_idx, height(region_features));
    
    % Optional: Display slice result for verification (every 5th slice)
    if slice_idx == 1 || mod(slice_idx, 5) == 0
        display_dwi_slice_result(original, brain_mask, interior_mask, artery_mask, slice_idx);
    end
end

% Close progress bar
close(h_waitbar);

% FINAL DATA CLEANUP
if height(all_features_table) > 0
    % Convert all string columns to cell arrays for Excel compatibility
    string_cols = {'Aneurysm', 'PatientGroup', 'StudyID', 'AcquisitionDate', ...
                   'Filename', 'RegionType', 'Modality'};
    for i = 1:length(string_cols)
        if ismember(string_cols{i}, all_features_table.Properties.VariableNames)
            all_features_table.(string_cols{i}) = cellstr(all_features_table.(string_cols{i}));
        end
    end
    
    % Ensure all numeric columns are double
    numeric_vars = varfun(@isnumeric, all_features_table, 'OutputFormat', 'uniform');
    numeric_cols = all_features_table.Properties.VariableNames(numeric_vars);
    for i = 1:length(numeric_cols)
        all_features_table.(numeric_cols{i}) = double(all_features_table.(numeric_cols{i}));
    end
end

% FINAL RESULTS SUMMARY
fprintf('\n%s\n', repmat('-', 1, 80));
fprintf('DWI BATCH PROCESSING COMPLETE: %d slices processed\n', n_slices);
fprintf('Total regions analyzed: %d (12 regions per slice, 3x4 grid)\n', height(all_features_table));
fprintf('Average regions per slice: %.1f\n', height(all_features_table)/n_slices);
fprintf('Modality: DWI (Diffusion-Weighted Imaging)\n');
fprintf('All patients are NORMAL - No aneurysms\n');
fprintf('%s\n', repmat('-', 1, 80));

% Display DWI-specific summary statistics
fprintf('\nDWI REGION-BASED ARTERY STATISTICS\n');
fprintf('  Mean artery density: %.2f%% ± %.2f%%\n', ...
    mean(all_features_table.artery_density), std(all_features_table.artery_density));
fprintf('  Mean artery count: %.1f ± %.1f\n', ...
    mean(all_features_table.num_arteries), std(all_features_table.num_arteries));
fprintf('  Mean artery area: %.1f ± %.1f pixels\n', ...
    mean(all_features_table.mean_area), std(all_features_table.mean_area));
fprintf('  Mean artery width: %.2f ± %.2f pixels\n', ...
    mean(all_features_table.mean_width), std(all_features_table.mean_width));
fprintf('  Mean artery intensity: %.3f ± %.3f\n', ...
    mean(all_features_table.mean_intensity), std(all_features_table.mean_intensity));

% SAVE RESULTS TO EXCEL
% Reorder columns to put key columns at the beginning
try
    all_cols = all_features_table.Properties.VariableNames;
    preferred_order = {'Aneurysm', 'PatientGroup', 'Modality', 'StudyID', 'AcquisitionDate', ...
                      'SliceNumber', 'RegionNumber', 'RegionRow', 'RegionCol', ...
                      'Filename', 'num_arteries', 'artery_density', 'mean_width', ...
                      'mean_aspect_ratio', 'mean_circularity', 'branch_point_density', ...
                      'mean_intensity'};
    
    existing_preferred = {};
    for i = 1:length(preferred_order)
        if ismember(preferred_order{i}, all_cols)
            existing_preferred{end+1} = preferred_order{i};
        end
    end
    
    remaining_cols = setdiff(all_cols, existing_preferred);
    all_features_table = all_features_table(:, [existing_preferred, remaining_cols]);
catch
    fprintf('Warning: Could not reorder columns. Using default order.\n');
end

% Save to Excel
output_filename = 'dwi_artery_features_12regions.xlsx';
try
    writetable(all_features_table, output_filename);
    fprintf('\nAll region features saved to: %s\n', output_filename);
catch ME
    fprintf('Error saving Excel file: %s\n', ME.message);
    csv_filename = 'dwi_artery_features_12regions.csv';
    writetable(all_features_table, csv_filename);
    fprintf('Saved as CSV instead: %s\n', csv_filename);
end

% Save workspace
mat_filename = 'dwi_artery_analysis_12regions.mat';
save(mat_filename, 'all_features_table', 'files', 'n_slices', 'patient_status');
fprintf('Complete workspace saved to: %s\n', mat_filename);

% Display preview
fprintf('\n%s\n', repmat('-', 1, 80));
fprintf('DWI REGION FEATURE TABLE (%d regions, %d features)\n', ...
    height(all_features_table), width(all_features_table));
fprintf('%s\n', repmat('-', 1, 80));

key_columns = {'RegionNumber', 'num_arteries', 'artery_density', 'mean_width', 'mean_intensity'};
existing_key_cols = {};
for i = 1:length(key_columns)
    if ismember(key_columns{i}, all_features_table.Properties.VariableNames)
        existing_key_cols{end+1} = key_columns{i};
    end
end

if ~isempty(existing_key_cols)
    disp(all_features_table(1:min(10, height(all_features_table)), existing_key_cols));
end

fprintf('\nAll %d regions labeled as DWI, ANEURYSM = "%s"\n', ...
    height(all_features_table), patient_status);

% DWI-SPECIFIC FUNCTIONS

% FUNCTION: DWI Wavelet Denoising
function denoised_img = wavelet_denoise_dwi(img, threshold_factor)
    if nargin < 2
        threshold_factor = 2.2;  % Higher for DWI noise
    end
    
    img_double = im2double(img);
    [coeffs, sizes] = wavedec2(img_double, 3, 'db4');
    
    % Estimate noise from finest detail coefficients
    detail_coeffs = coeffs(end-3*sizes(2,1)*sizes(2,2)+1:end);
    sigma = median(abs(detail_coeffs)) / 0.6745;
    threshold = threshold_factor * sigma;
    
    denoised_coeffs = coeffs;
    cA_length = prod(sizes(1, :));
    start_idx = cA_length + 1;
    
    for level = 1:3
        level_size = prod(sizes(level+1, :));
        end_idx = start_idx + 3*level_size - 1;
        
        for i = start_idx:end_idx
            if abs(coeffs(i)) > threshold
                denoised_coeffs(i) = sign(coeffs(i)) * (abs(coeffs(i)) - threshold);
            else
                denoised_coeffs(i) = 0;
            end
        end
        start_idx = end_idx + 1;
    end
    
    denoised = waverec2(denoised_coeffs, sizes, 'db4');
    denoised = mat2gray(denoised);
    denoised_img = im2double(denoised);
end

% FUNCTION: DWI Skull Stripping (Improved)
function brain_mask = skull_strip_dwi_improved(mri_image)
    img_double = im2double(mri_image);
    
    % DWI-specific thresholds (brain is brighter)
    level = graythresh(img_double);
    
    % Try multiple thresholds (adjusted for DWI)
    mask1 = imbinarize(img_double, level * 0.5);
    mask2 = imbinarize(img_double, level * 0.6);
    mask3 = imbinarize(img_double, level * 0.7);
    mask4 = imbinarize(img_double, level * 0.8);
    
    % Combine masks
    initial_mask = mask1 | mask2 | mask3 | mask4;
    
    % Clean up
    initial_mask = bwareaopen(initial_mask, 500);
    initial_mask = imfill(initial_mask, 'holes');
    
    % Select largest connected component (brain)
    CC = bwconncomp(initial_mask);
    
    if CC.NumObjects == 0
        % Fallback to intensity percentile
        sorted_vals = sort(img_double(:), 'descend');
        threshold_val = sorted_vals(round(0.3 * length(sorted_vals)));
        brain_mask = img_double > threshold_val * 0.6;
        brain_mask = bwareaopen(brain_mask, 1000);
        brain_mask = imfill(brain_mask, 'holes');
        
        CC = bwconncomp(brain_mask);
        if CC.NumObjects == 0
            [h, w] = size(img_double);
            [X, Y] = meshgrid(1:w, 1:h);
            brain_mask = ((X - w/2).^2 / (w/2.5)^2 + (Y - h/2).^2 / (h/2.5)^2) <= 1;
            return;
        end
    end
    
    numPixels = cellfun(@numel, CC.PixelIdxList);
    [~, idx] = max(numPixels);
    brain_mask = false(size(initial_mask));
    brain_mask(CC.PixelIdxList{idx}) = true;
    
    % Clean up edges
    se = strel('disk', 2);
    brain_mask = imclose(brain_mask, se);
    brain_mask = imfill(brain_mask, 'holes');
end

% FUNCTION: DWI Brain Interior Mask
function interior_mask = create_brain_interior_mask_dwi(brain_mask, erosion_pixels)
    if nargin < 2
        erosion_pixels = 4;  % Less erosion for DWI
    end
    
    se = strel('disk', erosion_pixels);
    interior_mask = imerode(brain_mask, se);
    
    fprintf('    Brain interior: %d pixels (%.1f%% of brain)\n', ...
        sum(interior_mask(:)), 100*sum(interior_mask(:))/sum(brain_mask(:)));
end

% FUNCTION: DWI Artery Extraction with 12 Regions
function [artery_mask, artery_map] = extract_dwi_arteries_12regions(img, brain_mask, interior_mask, num_regions)
    
    % Create 12 regions with full coverage
    [region_masks, blending_weights] = create_dwi_regions_12(interior_mask, 12, 12);
    
    if isempty(region_masks)
        artery_mask = false(size(img));
        artery_map = zeros(size(img));
        return;
    end
    
    num_valid_regions = length(region_masks);
    region_results = zeros([size(img), num_valid_regions]);
    
    % Analyze each region
    for r = 1:num_valid_regions
        region_mask = region_masks{r};
        
        % Extract region
        region_img = img;
        region_img(~region_mask) = 0;
        
        % DWI-specific Frangi for arteries (bright)
        region_frangi = frangi_filter_dwi_arteries(region_img, region_mask);
        
        % Adaptive threshold
        local_values = region_frangi(region_mask);
        if ~isempty(local_values) && sum(local_values > 0) > 5
            local_threshold = graythresh(local_values(local_values > 0)) * 0.9;
            region_result = region_frangi .* (region_frangi > local_threshold);
        else
            region_result = region_frangi;
        end
        
        region_results(:,:,r) = region_result .* region_mask;
    end
    
    % Blend results
    artery_map = zeros(size(img));
    for r = 1:num_valid_regions
        artery_map = artery_map + region_results(:,:,r) .* blending_weights(:,:,r);
    end
    
    % Normalize and threshold
    artery_map = mat2gray(artery_map);
    artery_map(~interior_mask) = 0;
    
    % Final threshold (conservative)
    interior_vals = artery_map(interior_mask);
    if ~isempty(interior_vals)
        opt_thresh = graythresh(interior_vals) * 1.1;  % Higher threshold
        artery_mask = artery_map > opt_thresh;
        artery_mask = artery_mask & interior_mask;
        artery_mask = bwareaopen(artery_mask, 8);
    else
        artery_mask = false(size(artery_map));
    end
end

% FUNCTION: Create DWI 12 Regions with Full Coverage
function [region_masks, blending_weights] = create_dwi_regions_12(interior_mask, num_regions, overlap)
    
    [rows, cols] = find(interior_mask);
    if isempty(rows)
        region_masks = {};
        blending_weights = [];
        return;
    end
    
    min_row = min(rows); max_row = max(rows);
    min_col = min(cols); max_col = max(cols);
    
    brain_height = max_row - min_row + 1;
    brain_width = max_col - min_col + 1;
    
    % Fixed 3x4 grid for 12 regions
    grid_rows = 3;
    grid_cols = 4;
    
    % Calculate base region size
    base_height = ceil(brain_height / grid_rows);
    base_width = ceil(brain_width / grid_cols);
    
    region_masks = {};
    region_centers = [];
    
    % Create regions with overlap
    for r = 1:grid_rows
        for c = 1:grid_cols
            % Calculate boundaries with overlap
            row_start = max(min_row, min_row + (r-1) * base_height - overlap);
            row_end = min(max_row, row_start + base_height - 1 + 2*overlap);
            col_start = max(min_col, min_col + (c-1) * base_width - overlap);
            col_end = min(max_col, col_start + base_width - 1 + 2*overlap);
            
            % Ensure last region reaches edge
            if r == grid_rows
                row_end = max_row;
            end
            if c == grid_cols
                col_end = max_col;
            end
            
            % Create region mask
            region_mask = false(size(interior_mask));
            region_mask(row_start:row_end, col_start:col_end) = true;
            region_mask = region_mask & interior_mask;
            
            if sum(region_mask(:)) > 150
                region_masks{end+1} = region_mask;
                [r_center, c_center] = find(region_mask);
                if ~isempty(r_center)
                    region_centers = [region_centers; mean(r_center), mean(c_center)];
                else
                    region_centers = [region_centers; (row_start+row_end)/2, (col_start+col_end)/2];
                end
            end
        end
    end
    
    % Create blending weights
    num_valid = length(region_masks);
    if num_valid == 0
        blending_weights = [];
        return;
    end
    
    [h, w] = size(interior_mask);
    [X, Y] = meshgrid(1:w, 1:h);
    
    blending_weights = zeros(h, w, num_valid);
    sigma = max(base_height, base_width) / 3;
    
    for r = 1:num_valid
        if r <= size(region_centers, 1)
            center_r = region_centers(r, 1);
            center_c = region_centers(r, 2);
            blending_weights(:,:,r) = exp(-((X - center_c).^2 + (Y - center_r).^2) / (2 * sigma^2));
            blending_weights(:,:,r) = blending_weights(:,:,r) .* region_masks{r};
        end
    end
    
    % Normalize
    weight_sum = sum(blending_weights, 3);
    weight_sum(weight_sum == 0) = 1;
    for r = 1:num_valid
        blending_weights(:,:,r) = blending_weights(:,:,r) ./ weight_sum;
    end
end

% FUNCTION: DWI Frangi Filter for Arteries (Bright Structures)
function vesselness = frangi_filter_dwi_arteries(img, brain_mask)
    scales = [1, 1.2, 1.5, 2, 2.5];  % Smaller scales for DWI
    Beta = 0.5; Gamma = 0.25;
    
    if ~isa(img, 'double')
        img = im2double(img);
    end
    
    % NO INVERSION - arteries are bright in DWI
    img_processed = img;
    
    vesselness = zeros(size(img));
    
    for s = 1:length(scales)
        sigma = scales(s);
        
        n = ceil(4*sigma);
        x = -n:n;
        y = -n:n;
        [X, Y] = meshgrid(x, y);
        
        G = exp(-(X.^2 + Y.^2) / (2*sigma^2));
        G = G / sum(G(:));
        
        Gxx = (X.^2 / sigma^4 - 1/sigma^2) .* G;
        Gxy = (X.*Y / sigma^4) .* G;
        Gyy = (Y.^2 / sigma^4 - 1/sigma^2) .* G;
        
        Dxx = conv2(img_processed, Gxx, 'same');
        Dxy = conv2(img_processed, Gxy, 'same');
        Dyy = conv2(img_processed, Gyy, 'same');
        
        for i = 1:size(img, 1)
            for j = 1:size(img, 2)
                if brain_mask(i, j)
                    H = [Dxx(i,j), Dxy(i,j); Dxy(i,j), Dyy(i,j)];
                    eigvals = eig(H);
                    lambda1 = max(eigvals);
                    lambda2 = min(eigvals);
                    
                    % For bright tubular structures (arteries)
                    if lambda2 < 0 && abs(lambda2) > 0
                        Rb = abs(lambda1) / (abs(lambda2) + eps);
                        S = sqrt(lambda1^2 + lambda2^2);
                        
                        response = exp(-Rb^2 / (2*Beta^2)) * (1 - exp(-S^2 / (2*Gamma^2)));
                        
                        % Penalize non-tubular structures
                        if abs(lambda1) < abs(lambda2) * 1.8
                            response = response * 0.5;
                        end
                        
                        vesselness(i,j) = max(vesselness(i,j), response);
                    end
                end
            end
        end
    end
    
    vesselness = imgaussfilt(vesselness, 0.5);
    vesselness = mat2gray(vesselness);
end

% FUNCTION: DWI Feature Extraction from 12 Regions
function region_features = extractArteryFeaturesDWI_12regions(artery_mask, interior_mask, brain_enhanced, ...
    slice_idx, filename, patient_status, patient_group, min_vessel_size)
    
    region_features = table();
    region_counter = 0;
    
    % Get brain interior bounding box
    [brain_rows, brain_cols] = find(interior_mask);
    if isempty(brain_rows)
        return;
    end
    min_row = min(brain_rows); max_row = max(brain_rows);
    min_col = min(brain_cols); max_col = max(brain_cols);
    
    % Fixed 3x4 grid for 12 regions
    grid_rows = 3;
    grid_cols = 4;
    
    % Calculate region size
    region_height = floor((max_row - min_row + 1) / grid_rows);
    region_width = floor((max_col - min_col + 1) / grid_cols);
    
    region_height = max(region_height, 35);
    region_width = max(region_width, 35);
    
    fprintf('    Grid: %d rows x %d columns, Region size: ~%d x %d pixels\n', ...
        grid_rows, grid_cols, region_height, region_width);
    
    % Generate grid regions
    for r = 1:grid_rows
        for c = 1:grid_cols
            region_counter = region_counter + 1;
            
            % Calculate region boundaries
            row_start = min_row + (r-1) * region_height;
            row_end = min(max_row, row_start + region_height - 1);
            col_start = min_col + (c-1) * region_width;
            col_end = min(max_col, col_start + region_width - 1);
            
            % Ensure last row/column reaches the edge
            if r == grid_rows
                row_end = max_row;
            end
            if c == grid_cols
                col_end = max_col;
            end
            
            % Create region mask (interior only)
            region_mask = false(size(interior_mask));
            region_mask(row_start:row_end, col_start:col_end) = true;
            region_mask = region_mask & interior_mask;
            
            % Skip if region too small
            if sum(region_mask(:)) < 100
                continue;
            end
            
            % Extract region-specific artery data
            region_arteries = artery_mask & region_mask;
            
            % Calculate DWI-specific features
            region_feat = extractDWIArteryFeatures(region_arteries, region_mask, brain_enhanced, min_vessel_size);
            
            % Add region metadata
            region_feat.SliceNumber = double(slice_idx);
            region_feat.RegionNumber = double(region_counter);
            region_feat.RegionRow = double(r);
            region_feat.RegionCol = double(c);
            region_feat.Filename = filename;
            region_feat.Aneurysm = patient_status;
            region_feat.PatientGroup = patient_group;
            region_feat.Modality = 'DWI';
            region_feat.StudyID = 'DWI_Study';
            region_feat.AcquisitionDate = datestr(now, 'yyyy-mm-dd');
            region_feat.RegionType = 'dwi_12region_grid';
            
            % Add region position metadata
            region_feat.RegionStartRow = double(row_start);
            region_feat.RegionEndRow = double(row_end);
            region_feat.RegionStartCol = double(col_start);
            region_feat.RegionEndCol = double(col_end);
            region_feat.RegionHeight = double(row_end - row_start + 1);
            region_feat.RegionWidth = double(col_end - col_start + 1);
            region_feat.RegionBrainPixels = double(sum(region_mask(:)));
            
            % Convert to table and append
            region_table = struct2table(region_feat);
            
            % Ensure consistent data types
            region_table.Filename = cellstr(region_table.Filename);
            region_table.Aneurysm = cellstr(region_table.Aneurysm);
            region_table.PatientGroup = cellstr(region_table.PatientGroup);
            region_table.Modality = cellstr(region_table.Modality);
            region_table.StudyID = cellstr(region_table.StudyID);
            region_table.AcquisitionDate = cellstr(region_table.AcquisitionDate);
            region_table.RegionType = cellstr(region_table.RegionType);
            
            % Append to results
            if isempty(region_features)
                region_features = region_table;
            else
                region_features = [region_features; region_table];
            end
        end
    end
    
    fprintf('    Generated %d DWI region samples\n', height(region_features));
end

% FUNCTION: DWI Artery Core Feature Extraction
function features = extractDWIArteryFeatures(artery_mask, brain_mask, brain_enhanced, min_vessel_size)
    features = struct();
    
    % 1. ARTERY COUNT & DENSITY (renamed from vessel to artery)
    [labeled, num_arteries] = bwlabel(artery_mask);
    features.num_arteries = num_arteries;
    brain_area_pixels = sum(brain_mask(:));
    artery_area_pixels = sum(artery_mask(:));
    features.artery_density = 100 * artery_area_pixels / brain_area_pixels;
    features.artery_area_pixels = artery_area_pixels;
    features.brain_area_pixels = brain_area_pixels;
    
    % If no arteries detected, set remaining features to NaN
    if num_arteries == 0
        features.mean_area = NaN;
        features.std_area = NaN;
        features.median_area = NaN;
        features.max_area = NaN;
        features.min_area = NaN;
        features.large_artery_count = 0;
        features.medium_artery_count = 0;
        features.small_artery_count = 0;
        features.tiny_artery_count = 0;
        features.small_artery_percentage = 0;
        features.mean_aspect_ratio = NaN;
        features.std_aspect_ratio = NaN;
        features.mean_eccentricity = NaN;
        features.std_eccentricity = NaN;
        features.mean_circularity = NaN;
        features.std_circularity = NaN;
        features.width_variability = NaN;
        features.mean_width = NaN;
        features.std_width = NaN;
        features.mean_intensity = NaN;
        features.median_intensity = NaN;
        features.intensity_std = NaN;
        features.intensity_range = NaN;
        features.glc_contrast = NaN;
        features.glc_correlation = NaN;
        features.glc_energy = NaN;
        features.glc_homogeneity = NaN;
        features.area_inequality = NaN;
        features.distance_to_center = NaN;
        features.total_artery_length = NaN;
        features.mean_artery_length = NaN;
        features.artery_complexity = NaN;
        features.branch_point_density = NaN;
        features.end_point_density = NaN;
        return;
    end
    
    % 2. SIZE DISTRIBUTION
    stats = regionprops(labeled, 'Area', 'MajorAxisLength', 'MinorAxisLength', ...
                       'Perimeter', 'Eccentricity', 'Centroid');
    areas = [stats.Area];
    
    features.mean_area = mean(areas);
    features.std_area = std(areas);
    features.median_area = median(areas);
    features.max_area = max(areas);
    features.min_area = min(areas);
    
    % Count by size categories (adjusted for arteries)
    features.large_artery_count = sum(areas >= 80);
    features.medium_artery_count = sum(areas >= 40 & areas < 80);
    features.small_artery_count = sum(areas >= 15 & areas < 40);
    features.tiny_artery_count = sum(areas < 15);
    features.small_artery_percentage = 100 * (features.tiny_artery_count + features.small_artery_count) / num_arteries;
    
    % 3. MORPHOLOGY & SHAPE
    aspect_ratios = zeros(1, num_arteries);
    circularities = zeros(1, num_arteries);
    widths = zeros(1, num_arteries);
    
    for i = 1:num_arteries
        if stats(i).MinorAxisLength > 0
            aspect_ratios(i) = stats(i).MajorAxisLength / stats(i).MinorAxisLength;
        else
            aspect_ratios(i) = NaN;
        end
        
        if stats(i).Perimeter > 0
            circularities(i) = 4 * pi * stats(i).Area / (stats(i).Perimeter^2);
        else
            circularities(i) = NaN;
        end
        
        widths(i) = stats(i).MinorAxisLength;
    end
    
    valid_aspect = aspect_ratios(~isnan(aspect_ratios) & aspect_ratios > 0);
    if ~isempty(valid_aspect)
        features.mean_aspect_ratio = mean(valid_aspect);
        features.std_aspect_ratio = std(valid_aspect);
    else
        features.mean_aspect_ratio = NaN;
        features.std_aspect_ratio = NaN;
    end
    
    features.mean_eccentricity = mean([stats.Eccentricity]);
    features.std_eccentricity = std([stats.Eccentricity]);
    
    valid_circular = circularities(~isnan(circularities) & circularities > 0);
    if ~isempty(valid_circular)
        features.mean_circularity = mean(valid_circular);
        features.std_circularity = std(valid_circular);
    else
        features.mean_circularity = NaN;
        features.std_circularity = NaN;
    end
    
    valid_widths = widths(widths > 0);
    if ~isempty(valid_widths)
        features.width_variability = std(valid_widths) / (mean(valid_widths) + eps);
        features.mean_width = mean(valid_widths);
        features.std_width = std(valid_widths);
    else
        features.width_variability = NaN;
        features.mean_width = NaN;
        features.std_width = NaN;
    end
    
    % 4. INTENSITY CHARACTERISTICS (DWI-specific)
    artery_intensities = brain_enhanced(artery_mask);
    features.mean_intensity = mean(artery_intensities);
    features.median_intensity = median(artery_intensities);
    features.intensity_std = std(artery_intensities);
    features.intensity_range = range(artery_intensities);
    
    % 5. TEXTURE FEATURES (GLCM)
    if any(artery_mask(:))
        [y, x] = find(artery_mask);
        y1 = max(1, min(y)-8);
        y2 = min(size(brain_enhanced,1), max(y)+8);
        x1 = max(1, min(x)-8);
        x2 = min(size(brain_enhanced,2), max(x)+8);
        
        roi = brain_enhanced(y1:y2, x1:x2);
        if numel(roi) > 1
            glcm = graycomatrix(roi, 'NumLevels', 8, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
            glcm_stats = graycoprops(glcm);
            features.glc_contrast = mean(glcm_stats.Contrast);
            features.glc_correlation = mean(glcm_stats.Correlation);
            features.glc_energy = mean(glcm_stats.Energy);
            features.glc_homogeneity = mean(glcm_stats.Homogeneity);
        else
            features.glc_contrast = NaN;
            features.glc_correlation = NaN;
            features.glc_energy = NaN;
            features.glc_homogeneity = NaN;
        end
    else
        features.glc_contrast = NaN;
        features.glc_correlation = NaN;
        features.glc_energy = NaN;
        features.glc_homogeneity = NaN;
    end
    
    % 6. SPATIAL DISTRIBUTION
    sorted_areas = sort(areas);
    n = length(sorted_areas);
    if n > 1
        cumulative = cumsum(sorted_areas);
        perfect_line = (1:n) * (sum(areas) / n);
        features.area_inequality = sum(abs(cumulative - perfect_line)) / (sum(areas) * n + eps);
    else
        features.area_inequality = 0;
    end
    
    centroids = vertcat(stats.Centroid);
    if ~isempty(centroids) && size(centroids, 1) > 0
        [h, w] = size(brain_mask);
        mean_centroid = mean(centroids, 1);
        features.distance_to_center = sqrt((mean_centroid(1) - w/2)^2 + ...
                                          (mean_centroid(2) - h/2)^2) / sqrt((w/2)^2 + (h/2)^2);
    else
        features.distance_to_center = NaN;
    end
    
    % 7. ARTERY NETWORK FEATURES
    artery_skeleton = bwmorph(artery_mask, 'skel', Inf);
    features.total_artery_length = sum(artery_skeleton(:));
    features.mean_artery_length = features.total_artery_length / num_arteries;
    
    branch_points = bwmorph(artery_skeleton, 'branchpoints');
    end_points = bwmorph(artery_skeleton, 'endpoints');
    
    features.branch_point_density = 100 * sum(branch_points(:)) / (artery_area_pixels + eps);
    features.end_point_density = 100 * sum(end_points(:)) / (artery_area_pixels + eps);
    
    if features.total_artery_length > 0
        features.artery_complexity = sum(branch_points(:)) / features.total_artery_length;
    else
        features.artery_complexity = 0;
    end
end

% FUNCTION: Ensure Consistent Data Types
function new_table = ensureConsistentTypes(new_table, master_table)
    master_vars = master_table.Properties.VariableNames;
    
    % Add any missing variables to new_table
    missing_in_new = setdiff(master_vars, new_table.Properties.VariableNames);
    for i = 1:length(missing_in_new)
        var_name = missing_in_new{i};
        if isnumeric(master_table.(var_name))
            new_table.(var_name) = NaN(height(new_table), 1);
        elseif islogical(master_table.(var_name))
            new_table.(var_name) = false(height(new_table), 1);
        elseif iscell(master_table.(var_name))
            new_table.(var_name) = cell(height(new_table), 1);
        else
            new_table.(var_name) = repmat({''}, height(new_table), 1);
        end
    end
    
    % Reorder columns to match master table
    new_table = new_table(:, master_vars);
end

% FUNCTION: Display DWI Slice Result
function display_dwi_slice_result(original, brain_mask, interior_mask, artery_mask, slice_idx)
    figure('Name', sprintf('DWI Slice %d Verification', slice_idx), ...
           'Position', [100, 100, 1200, 400]);
    
    % Original
    subplot(1,3,1);
    imshow(original, []);
    title(sprintf('DWI Slice %d: Original', slice_idx), 'FontSize', 10);
    
    % Brain interior (green) vs suppressed edge (red)
    subplot(1,3,2);
    imshow(original, []);
    hold on;
    interior_outline = bwperim(interior_mask);
    [ry, rx] = find(interior_outline);
    plot(rx, ry, 'g.', 'MarkerSize', 1);
    
    edge_mask = brain_mask & ~interior_mask;
    edge_outline = bwperim(edge_mask);
    [ry, rx] = find(edge_outline);
    plot(rx, ry, 'r.', 'MarkerSize', 1);
    title('Green: Interior, Red: Suppressed Edge', 'FontSize', 10);
    hold off;
    
    % Arteries (only in interior)
    subplot(1,3,3);
    overlay = repmat(im2double(original), [1,1,3]);
    artery_outline = bwperim(artery_mask);
    for c = 1:3
        channel = overlay(:,:,c);
        if c == 1 % Red
            channel(artery_outline) = 1;
        else
            channel(artery_outline) = 0;
        end
        overlay(:,:,c) = channel;
    end
    imshow(overlay);
    title(sprintf('Arteries (Red) - Count: %d', bwconncomp(artery_mask).NumObjects), 'FontSize', 10);
    
    % Save figure
    saveas(gcf, sprintf('dwi_verification_slice_%d.png', slice_idx));
    close(gcf);
end