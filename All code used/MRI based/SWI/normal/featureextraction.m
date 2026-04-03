% MAIN SCRIPT - REGION-BASED BLOOD VESSEL SEGMENTATION WITH FEATURE EXTRACTION

clear all; clc; close all;

% IMAGE LOADING AND BATCH PROCESSING
% Load ALL images from current directory
files = dir('*.jpg');
n_slices = length(files);
fprintf('Found %d image slices to process\n', n_slices);

if n_slices == 0
    error('No JPG images found in current directory');
end

% Initialize master table as empty
all_features_table = table();

% Create progress bar
h_waitbar = waitbar(0, 'Processing MRI slices...');

% Define patient status - NORMAL (no aneurysm)
patient_status = 'no';
patient_group = 'Normal';

% OPTIMIZED REGION ANALYSIS PARAMETERS - 12 NON-OVERLAPPING REGIONS
% Using 12 non-overlapping regions (3x4 grid) with overlap for smooth blending
target_regions = 12; % 12 regions per slice (3x4 grid)
region_overlap = 10; % 10 pixels overlap to prevent seam artifacts

fprintf('\nOPTIMIZED REGION ANALYSIS CONFIGURATION:\n');
fprintf('  Regions per slice: %d (3x4 grid with %d-pixel overlap)\n', target_regions, region_overlap);
fprintf('  Edge suppression: 8-pixel erosion of brain boundary\n');
fprintf('  Total regions to process: %d\n', target_regions * n_slices);

% Process each slice
for slice_idx = 1:n_slices
    % Update progress
    waitbar(slice_idx/n_slices, h_waitbar, sprintf('Processing slice %d/%d: %s', ...
        slice_idx, n_slices, files(slice_idx).name));
    
    fprintf('\n%s\n', repmat('-', 1, 80));
    fprintf('PROCESSING SLICE %d/%d: %s\n', slice_idx, n_slices, files(slice_idx).name);
    fprintf('%s\n', repmat('-', 1, 80));
    
    % Load and preprocess current slice
    img = imread(files(slice_idx).name);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    original = im2double(img);

    % PREPROCESSING PIPELINE (WITH EDGE SUPPRESSION)
    fprintf('\nStep 1: Preprocessing with edge suppression...\n');
    
    % Wavelet denoising
    brain_denoised = wavelet_denoise_swi(original);
    
    % Skull stripping (multi-threshold)
    brain_mask = skull_strip_swi_improved(brain_denoised);
    
    % Create brain interior mask (remove edge artifacts)
    interior_mask = create_brain_interior_mask(brain_mask, 8);
    
    % Apply mask to get brain only
    brain_only = brain_denoised .* brain_mask;
    
    % Global CLAHE enhancement
    brain_enhanced = adapthisteq(brain_only, ...
        'NumTiles', [8 8], ...
        'ClipLimit', 0.02, ...
        'Distribution', 'rayleigh', ...
        'Alpha', 0.5);
    brain_enhanced(~brain_mask) = 0;

    % BLENDED REGION ANALYSIS FOR VESSEL SEGMENTATION (WITH EDGE SUPPRESSION)
    fprintf('\nStep 2: Performing region analysis with %d regions (edge-suppressed)...\n', target_regions);
    
    % Use the improved vessel extraction with edge suppression
    [vessel_mask, vesselness_map] = extract_brain_vessels_improved(brain_enhanced, brain_mask, interior_mask, target_regions);

    % REGION-BASED FEATURE EXTRACTION - 12 NON-OVERLAPPING REGIONS
    fprintf('\nStep 3: Extracting features from %d non-overlapping regions (interior only)...\n', target_regions);
    
    % Extract features from each region using interior mask (no edge artifacts)
    region_features = extractFeaturesNonOverlapping12_improved(vessel_mask, interior_mask, brain_enhanced, ...
        slice_idx, files(slice_idx).name, patient_status, patient_group);
    
    % Add to master table
    if isempty(all_features_table)
        all_features_table = region_features;
    else
        % Ensure consistent variable types before concatenation
        region_features = ensureConsistentTypes(region_features, all_features_table);
        all_features_table = [all_features_table; region_features];
    end
    
    fprintf('\nSlice %d complete - Generated %d region samples (edge-suppressed)\n', ...
        slice_idx, height(region_features));
    
    % Optional: Display slice result for verification
    if slice_idx == 1 || mod(slice_idx, 10) == 0
        display_slice_result(original, brain_mask, interior_mask, vessel_mask, slice_idx);
    end
end

% Close progress bar
close(h_waitbar);

% FINAL DATA CLEANUP
if height(all_features_table) > 0
    % Convert all string columns to cell arrays for Excel compatibility
    string_cols = {'Aneurysm', 'PatientGroup', 'StudyID', 'AcquisitionDate', ...
                   'Filename', 'RegionType'};
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
fprintf('BATCH PROCESSING COMPLETE: %d slices processed\n', n_slices);
fprintf('Total regions analyzed: %d (12 regions per slice, 3x4 grid)\n', height(all_features_table));
fprintf('Average regions per slice: %.1f\n', height(all_features_table)/n_slices);
fprintf('Edge suppression: 8-pixel erosion applied to all slices\n');
fprintf('All patients are NORMAL - No aneurysms detected\n');
fprintf('%s\n', repmat('-', 1, 80));

% Display summary statistics
fprintf('\nREGION-BASED SUMMARY STATISTICS (12 Non-Overlapping Regions, Edge-Suppressed)\n');
fprintf('  Mean vessel density: %.2f%% ± %.2f%%\n', ...
    mean(all_features_table.vessel_density), std(all_features_table.vessel_density));
fprintf('  Mean vessel count: %.1f ± %.1f\n', ...
    mean(all_features_table.num_vessels), std(all_features_table.num_vessels));
fprintf('  Mean vessel area: %.1f ± %.1f pixels\n', ...
    mean(all_features_table.mean_area), std(all_features_table.mean_area));
fprintf('  Mean vessel width: %.2f ± %.2f pixels\n', ...
    mean(all_features_table.mean_width), std(all_features_table.mean_width));

% SAVE RESULTS TO EXCEL
% Reorder columns to put key columns at the beginning
try
    all_cols = all_features_table.Properties.VariableNames;
    preferred_order = {'Aneurysm', 'PatientGroup', 'StudyID', 'AcquisitionDate', ...
                      'SliceNumber', 'RegionNumber', 'RegionRow', 'RegionCol', ...
                      'Filename', 'num_vessels', 'vessel_density', 'mean_width', ...
                      'mean_aspect_ratio', 'mean_circularity', 'branch_point_density'};
    
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

% Save to Excel and CSV
output_filename = 'normal_patients_12region_features_edgesuppressed.xlsx';
try
    writetable(all_features_table, output_filename);
    fprintf('\nAll region features saved to: %s\n', output_filename);
catch ME
    fprintf('Error saving Excel file: %s\n', ME.message);
    csv_filename = 'normal_patients_12region_features_edgesuppressed.csv';
    writetable(all_features_table, csv_filename);
    fprintf('Saved as CSV instead: %s\n', csv_filename);
end

% Save workspace
mat_filename = 'normal_patients_12region_analysis_edgesuppressed.mat';
save(mat_filename, 'all_features_table', 'files', 'n_slices', 'patient_status');
fprintf('Complete workspace saved to: %s\n', mat_filename);

% Display preview
fprintf('\n%s\n', repmat('-', 1, 80));
fprintf('REGION FEATURE TABLE (%d regions, %d features)\n', ...
    height(all_features_table), width(all_features_table));
fprintf('%s\n', repmat('-', 1, 80));

key_columns = {'Aneurysm', 'RegionNumber', 'RegionRow', 'RegionCol', 'num_vessels', 'vessel_density'};
existing_key_cols = {};
for i = 1:length(key_columns)
    if ismember(key_columns{i}, all_features_table.Properties.VariableNames)
        existing_key_cols{end+1} = key_columns{i};
    end
end

if ~isempty(existing_key_cols)
    disp(all_features_table(1:min(10, height(all_features_table)), existing_key_cols));
end

fprintf('\nAll %d regions have been labeled as: ANEURYSM = "%s"\n', ...
    height(all_features_table), patient_status);

% ALL FUNCTIONS DEFINED AT THE END OF FILE

% FUNCTION: Skull Stripping (Multi-threshold)
function brain_mask = skull_strip_swi_improved(mri_image)
    img_double = im2double(mri_image);
    
    % Multi-level thresholding for better brain extraction
    level = graythresh(img_double);
    
    % Try multiple thresholds and combine
    mask1 = imbinarize(img_double, level * 0.6);
    mask2 = imbinarize(img_double, level * 0.7);
    mask3 = imbinarize(img_double, level * 0.8);
    
    % Combine masks
    initial_mask = mask1 | mask2 | mask3;
    
    % Clean up
    initial_mask = bwareaopen(initial_mask, 500);
    initial_mask = imfill(initial_mask, 'holes');
    
    % Select largest connected component (brain)
    CC = bwconncomp(initial_mask);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    [~, idx] = max(numPixels);
    brain_mask = false(size(initial_mask));
    brain_mask(CC.PixelIdxList{idx}) = true;
    
    % Erode slightly to remove any remaining skull/edge
    se = strel('disk', 3);
    brain_mask = imerode(brain_mask, se);
    
    % Clean up edges
    brain_mask = imfill(brain_mask, 'holes');
    brain_mask = imclose(brain_mask, strel('disk', 5));
    
    % Remove small protrusions
    brain_mask = bwmorph(brain_mask, 'spur', 5);
    
    % Final smoothing
    brain_mask = imdilate(brain_mask, strel('disk', 2));
end

% FUNCTION: Create Brain Interior Mask (Removes Edge Artifacts)
function interior_mask = create_brain_interior_mask(brain_mask, erosion_pixels)
    if nargin < 2
        erosion_pixels = 8;
    end
    
    % Erode brain mask to get interior (away from edge)
    se = strel('disk', erosion_pixels);
    interior_mask = imerode(brain_mask, se);
    
    fprintf('    Brain interior: %d pixels (%.1f%% of brain)\n', ...
        sum(interior_mask(:)), 100*sum(interior_mask(:))/sum(brain_mask(:)));
end

% FUNCTION: Vessel Extraction (No Edge Artifacts)
function [vessel_mask, vesselness_map] = extract_brain_vessels_improved(img, brain_mask, interior_mask, num_regions)
    
    % Analyze with regions (interior only)
    vesselness_map = analyze_regions_improved(img, interior_mask, num_regions);
    
    % Normalize
    vesselness_map = mat2gray(vesselness_map);
    vesselness_map(~interior_mask) = 0;
    
    % Adaptive thresholding on interior only
    interior_values = vesselness_map(interior_mask);
    if ~isempty(interior_values)
        % Use Otsu's method on interior only
        opt_thresh = graythresh(interior_values);
        
        % Hysteresis thresholds
        low_thresh = opt_thresh * 0.7;
        high_thresh = opt_thresh * 1.3;
        
        % Strong vessels
        strong_vessels = vesselness_map > high_thresh;
        
        % Weak vessels
        weak_vessels = vesselness_map > low_thresh & vesselness_map <= high_thresh;
        
        % Connect weak to strong
        CC = bwconncomp(weak_vessels);
        weak_connected = false(size(weak_vessels));
        
        for i = 1:CC.NumObjects
            component = false(size(weak_vessels));
            component(CC.PixelIdxList{i}) = true;
            
            strong_dilated = imdilate(strong_vessels, strel('disk', 3));
            if any(component(:) & strong_dilated(:))
                weak_connected(CC.PixelIdxList{i}) = true;
            end
        end
        
        % Combine
        vessel_mask = strong_vessels | weak_connected;
        
        % Clean up
        vessel_mask = bwareaopen(vessel_mask, 15);
        vessel_mask = imclose(vessel_mask, strel('disk', 1));
        vessel_mask = vessel_mask & interior_mask;
    else
        vessel_mask = false(size(vesselness_map));
    end
end

% FUNCTION: Analyze Regions (With Edge Suppression)
function blended_vesselness = analyze_regions_improved(img, interior_mask, num_regions)
    
    % Split brain interior into regions with overlap
    [region_masks, blending_weights] = create_regions_improved(interior_mask, num_regions, 10);
    num_valid_regions = length(region_masks);
    
    if num_valid_regions == 0
        blended_vesselness = zeros(size(img));
        return;
    end
    
    % Initialize vesselness results
    region_results = zeros([size(img), num_valid_regions]);
    
    % Analyze each region
    for r = 1:num_valid_regions
        region_mask = region_masks{r};
        
        % Create region-specific image
        region_img = img;
        region_img(~region_mask) = 0;
        
        % Apply Frangi with edge suppression built in
        region_frangi = frangi_vesselness_swi(region_img);
        
        % Apply region-specific threshold
        local_values = region_frangi(region_mask);
        if ~isempty(local_values) && sum(local_values > 0) > 10
            local_threshold = graythresh(local_values(local_values > 0)) * 0.85;
            region_result = region_frangi .* (region_frangi > local_threshold);
        else
            region_result = region_frangi;
        end
        
        region_results(:,:,r) = region_result .* region_mask;
    end
    
    % Blend results
    blended_vesselness = zeros(size(img));
    for r = 1:num_valid_regions
        blended_vesselness = blended_vesselness + region_results(:,:,r) .* blending_weights(:,:,r);
    end
    
    % Apply smoothing
    blended_vesselness = imgaussfilt(blended_vesselness, 1.0);
    
    if max(blended_vesselness(:)) > 0
        blended_vesselness = blended_vesselness / max(blended_vesselness(:));
    end
    blended_vesselness(~interior_mask) = 0;
end

% FUNCTION: Create Regions (With Overlap)
function [region_masks, blending_weights] = create_regions_improved(interior_mask, num_regions, overlap)
    
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
    if num_regions == 12
        grid_rows = 3;
        grid_cols = 4;
    else
        % Fallback for other numbers
        grid_cols = round(sqrt(num_regions));
        grid_rows = ceil(num_regions / grid_cols);
    end
    
    % Calculate base region size
    base_height = floor(brain_height / grid_rows);
    base_width = floor(brain_width / grid_cols);
    
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
            
            % Create region mask
            region_mask = false(size(interior_mask));
            region_mask(row_start:row_end, col_start:col_end) = true;
            region_mask = region_mask & interior_mask;
            
            if sum(region_mask(:)) > 200
                region_masks{end+1} = region_mask;
                region_centers = [region_centers; (row_start+row_end)/2, (col_start+col_end)/2];
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
        center_r = region_centers(r, 1);
        center_c = region_centers(r, 2);
        blending_weights(:,:,r) = exp(-((X - center_c).^2 + (Y - center_r).^2) / (2 * sigma^2));
        blending_weights(:,:,r) = blending_weights(:,:,r) .* region_masks{r};
    end
    
    % Normalize
    weight_sum = sum(blending_weights, 3);
    weight_sum(weight_sum == 0) = 1;
    for r = 1:num_valid
        blending_weights(:,:,r) = blending_weights(:,:,r) ./ weight_sum;
    end
end

% FUNCTION: Extract Features from 12 Non-Overlapping Regions (3x4 Grid)
function region_features = extractFeaturesNonOverlapping12_improved(vessel_mask, interior_mask, brain_enhanced, ...
    slice_idx, filename, patient_status, patient_group)
    
    region_features = table();
    region_counter = 0;
    [img_h, img_w] = size(interior_mask);
    
    % Get brain interior bounding box
    [brain_rows, brain_cols] = find(interior_mask);
    if isempty(brain_rows)
        return;
    end
    min_row = min(brain_rows); max_row = max(brain_rows);
    min_col = min(brain_cols); max_col = max(brain_cols);
    
    % Fixed 3x4 grid for 12 non-overlapping regions
    grid_rows = 3;
    grid_cols = 4;
    
    % Calculate region size (non-overlapping)
    region_height = floor((max_row - min_row + 1) / grid_rows);
    region_width = floor((max_col - min_col + 1) / grid_cols);
    
    % Ensure minimum region size
    region_height = max(region_height, 40);
    region_width = max(region_width, 40);
    
    fprintf('    Grid: %d rows x %d columns, Region size: ~%d x %d pixels (interior only)\n', ...
        grid_rows, grid_cols, region_height, region_width);
    
    % Generate non-overlapping grid regions
    for r = 1:grid_rows
        for c = 1:grid_cols
            region_counter = region_counter + 1;
            
            % Calculate region boundaries (non-overlapping)
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
            
            % Create region mask (restricted to interior only)
            region_mask = false(size(interior_mask));
            region_mask(row_start:row_end, col_start:col_end) = true;
            region_mask = region_mask & interior_mask;  % Only interior pixels
            
            % Skip if region has too little brain tissue
            if sum(region_mask(:)) < 100
                continue;
            end
            
            % Extract region-specific vessel and brain data
            region_vessels = vessel_mask & region_mask;  % Vessels only in interior
            
            % Calculate region-specific features
            region_feat = extractCoreFeatures(region_vessels, region_mask, brain_enhanced);
            
            % Add region metadata
            region_feat.SliceNumber = double(slice_idx);
            region_feat.RegionNumber = double(region_counter);
            region_feat.RegionRow = double(r);
            region_feat.RegionCol = double(c);
            region_feat.Filename = filename;
            region_feat.Aneurysm = patient_status;
            region_feat.PatientGroup = patient_group;
            region_feat.StudyID = 'Normal_Baseline';
            region_feat.AcquisitionDate = datestr(now, 'yyyy-mm-dd');
            region_feat.RegionType = 'non-overlapping_3x4_interior';
            
            % Add region position metadata
            region_feat.RegionStartRow = double(row_start);
            region_feat.RegionEndRow = double(row_end);
            region_feat.RegionStartCol = double(col_start);
            region_feat.RegionEndCol = double(col_end);
            region_feat.RegionHeight = double(row_end - row_start + 1);
            region_feat.RegionWidth = double(col_end - col_start + 1);
            region_feat.RegionBrainPixels = double(sum(region_mask(:)));
            region_feat.RegionBrainPercentage = 100 * sum(region_mask(:)) / ((row_end-row_start+1)*(col_end-col_start+1));
            
            % Convert to table and append
            region_table = struct2table(region_feat);
            
            % Ensure consistent data types
            region_table.Filename = cellstr(region_table.Filename);
            region_table.Aneurysm = cellstr(region_table.Aneurysm);
            region_table.PatientGroup = cellstr(region_table.PatientGroup);
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
    
    fprintf('    Generated %d region samples (interior only, edge-suppressed)\n', height(region_features));
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

% FUNCTION: Core Feature Extraction
function features = extractCoreFeatures(vessel_mask, brain_mask, brain_enhanced)
    features = struct();
    
    % 1. VESSEL COUNT & DENSITY
    [labeled, num_vessels] = bwlabel(vessel_mask);
    features.num_vessels = num_vessels;
    brain_area_pixels = sum(brain_mask(:));
    vessel_area_pixels = sum(vessel_mask(:));
    features.vessel_density = 100 * vessel_area_pixels / brain_area_pixels;
    features.vessel_area_pixels = vessel_area_pixels;
    features.brain_area_pixels = brain_area_pixels;
    
    % If no vessels detected, set remaining features to NaN/0
    if num_vessels == 0
        features.mean_area = NaN;
        features.std_area = NaN;
        features.median_area = NaN;
        features.max_area = NaN;
        features.min_area = NaN;
        features.large_vessel_count = 0;
        features.medium_vessel_count = 0;
        features.small_vessel_count = 0;
        features.tiny_vessel_count = 0;
        features.small_vessel_percentage = 0;
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
        features.total_vessel_length = NaN;
        features.mean_vessel_length = NaN;
        features.vessel_complexity = NaN;
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
    
    % Count by size categories
    features.large_vessel_count = sum(areas >= 100);
    features.medium_vessel_count = sum(areas >= 50 & areas < 100);
    features.small_vessel_count = sum(areas >= 20 & areas < 50);
    features.tiny_vessel_count = sum(areas < 20);
    features.small_vessel_percentage = 100 * (features.tiny_vessel_count + features.small_vessel_count) / num_vessels;
    
    % 3. MORPHOLOGY & SHAPE
    aspect_ratios = zeros(1, num_vessels);
    circularities = zeros(1, num_vessels);
    widths = zeros(1, num_vessels);
    
    for i = 1:num_vessels
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
    
    % 4. INTENSITY CHARACTERISTICS
    vessel_intensities = brain_enhanced(vessel_mask);
    features.mean_intensity = mean(vessel_intensities);
    features.median_intensity = median(vessel_intensities);
    features.intensity_std = std(vessel_intensities);
    features.intensity_range = range(vessel_intensities);
    
    % 5. TEXTURE FEATURES (GLCM)
    if any(vessel_mask(:))
        [y, x] = find(vessel_mask);
        y1 = max(1, min(y)-10);
        y2 = min(size(brain_enhanced,1), max(y)+10);
        x1 = max(1, min(x)-10);
        x2 = min(size(brain_enhanced,2), max(x)+10);
        
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
    
    % 7. VESSEL NETWORK FEATURES
    vessel_skeleton = bwmorph(vessel_mask, 'skel', Inf);
    features.total_vessel_length = sum(vessel_skeleton(:));
    features.mean_vessel_length = features.total_vessel_length / num_vessels;
    
    branch_points = bwmorph(vessel_skeleton, 'branchpoints');
    end_points = bwmorph(vessel_skeleton, 'endpoints');
    
    features.branch_point_density = 100 * sum(branch_points(:)) / (vessel_area_pixels + eps);
    features.end_point_density = 100 * sum(end_points(:)) / (vessel_area_pixels + eps);
    
    if features.total_vessel_length > 0
        features.vessel_complexity = sum(branch_points(:)) / features.total_vessel_length;
    else
        features.vessel_complexity = 0;
    end
end

% FUNCTION: Wavelet Denoising
function denoised_img = wavelet_denoise_swi(img)
    img_double = im2double(img);
    [coeffs, sizes] = wavedec2(img_double, 3, 'db4');
    
    detail_coeffs = coeffs(end-3*sizes(2,1)*sizes(2,2)+1:end);
    sigma = median(abs(detail_coeffs)) / 0.6745;
    threshold = 2.1 * sigma;
    
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
    
    if isa(img, 'uint8')
        denoised_img = im2uint8(denoised);
    else
        denoised_img = denoised;
    end
end

% FUNCTION: Frangi Vesselness Filter
function vesselness = frangi_vesselness_swi(img)
    scales = [1.5, 2, 2.5, 3, 3.5, 4];
    Beta = 0.52; Gamma = 0.26;
    
    if ~isa(img, 'double')
        img = im2double(img);
    end
    img_processed = 1 - img;
    
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
        
        lambda1 = zeros(size(img));
        lambda2 = zeros(size(img));
        
        for i = 1:size(img, 1)
            for j = 1:size(img, 2)
                H = [Dxx(i,j), Dxy(i,j); Dxy(i,j), Dyy(i,j)];
                eigvals = eig(H);
                lambda1(i,j) = max(eigvals);
                lambda2(i,j) = min(eigvals);
            end
        end
        
        Rb = abs(lambda1) ./ (abs(lambda2) + eps);
        S = sqrt(lambda1.^2 + lambda2.^2);
        
        response = exp(-Rb.^2 / (2*Beta^2)) .* (1 - exp(-S.^2 / (2*Gamma^2)));
        response(lambda2 >= -0.02) = 0;
        
        vesselness = max(vesselness, response);
    end
    
    vesselness = imgaussfilt(vesselness, 0.5);
    vesselness = mat2gray(vesselness);
end

% FUNCTION: Display Slice Result (for verification)
function display_slice_result(original, brain_mask, interior_mask, vessel_mask, slice_idx)
    figure('Name', sprintf('Slice %d Verification', slice_idx), ...
           'Position', [100, 100, 1200, 400], ...
           'Visible', 'off');
    
    % Original
    subplot(1,3,1);
    imshow(original, []);
    title(sprintf('Slice %d: Original', slice_idx), 'FontSize', 10);
    
    % Brain interior (green) vs suppressed edge (red)
    subplot(1,3,2);
    imshow(original, []);
    hold on;
    % Show interior in green
    interior_outline = bwperim(interior_mask);
    [ry, rx] = find(interior_outline);
    plot(rx, ry, 'g.', 'MarkerSize', 1);
    % Show suppressed edge in red
    edge_mask = brain_mask & ~interior_mask;
    edge_outline = bwperim(edge_mask);
    [ry, rx] = find(edge_outline);
    plot(rx, ry, 'r.', 'MarkerSize', 1);
    title('Green: Interior, Red: Suppressed Edge', 'FontSize', 10);
    hold off;
    
    % Vessels (only in interior)
    subplot(1,3,3);
    % Create RGB overlay
    overlay = repmat(im2double(original), [1,1,3]);
    % Show vessels in red (only where interior)
    vessel_outline = bwperim(vessel_mask);
    for c = 1:3
        channel = overlay(:,:,c);
        if c == 1 % Red
            channel(vessel_outline) = 1;
        else
            channel(vessel_outline) = 0;
        end
        overlay(:,:,c) = channel;
    end
    imshow(overlay);
    title('Vessels (Red) in Interior Only', 'FontSize', 10);
    
    % Save figure
    saveas(gcf, sprintf('verification_slice_%d.png', slice_idx));
    close(gcf);
end