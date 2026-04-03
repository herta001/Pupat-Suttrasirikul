% DWI BRAIN ARTERY EXTRACTION - 12 REGIONS WITH FULL BRAIN COVERAGE

clear all; clc; close all;

% RANDOM IMAGE SELECTION
files = [dir('*.jpg'); dir('*.png'); dir('*.tif'); dir('*.bmp'); dir('*.dcm')];
if isempty(files)
    error('No image files found. Place DWI images in current folder.');
end

% Pick random image
random_idx = randi(length(files));
img = imread(files(random_idx).name);

fprintf('\n%s\n', repmat('-', 1, 70));
fprintf('DWI ARTERY EXTRACTION - 12 REGIONS (FULL COVERAGE)\n');
fprintf('Random Slice %d/%d: %s\n', random_idx, length(files), files(random_idx).name);
fprintf('%s\n', repmat('-', 1, 70));

if size(img, 3) == 3
    original = rgb2gray(img);
else
    original = img;
end
original = im2double(original);

% ARTERY-SPECIFIC PARAMETERS
params = struct(...
    'frangi_scales', [1, 1.2, 1.5, 2], ...      % Scales for arteries
    'frangi_beta', 0.5, ...
    'frangi_gamma', 0.25, ...
    'edge_erosion', 4, ...                        % Slight erosion
    'wavelet_threshold', 2.0, ...
    'cliplimit', 0.02, ...
    'min_vessel_size', 12, ...                    % Minimum artery size
    'vessel_threshold_factor', 1.05, ...          % Threshold
    'edge_suppression_strength', 0.1 ...          % Edge suppression
);

% FUNCTION: Skull Stripping (Preserves Brain Shape)
function brain_mask = skull_strip_dwi(mri_image)
    img_double = im2double(mri_image);
    
    % Multi-level thresholding to capture full brain
    level = graythresh(img_double);
    thresholds = [0.4, 0.5, 0.6, 0.7];
    combined_mask = false(size(img_double));
    
    for t = 1:length(thresholds)
        current_mask = imbinarize(img_double, level * thresholds(t));
        current_mask = bwareaopen(current_mask, 200);
        combined_mask = combined_mask | current_mask;
    end
    
    % Fill holes and clean
    combined_mask = imfill(combined_mask, 'holes');
    combined_mask = bwareaopen(combined_mask, 2000);
    
    % Find largest component (brain)
    CC = bwconncomp(combined_mask);
    
    if CC.NumObjects > 0
        numPixels = cellfun(@numel, CC.PixelIdxList);
        [~, idx] = max(numPixels);
        brain_mask = false(size(img_double));
        brain_mask(CC.PixelIdxList{idx}) = true;
    else
        % Fallback to intensity percentile
        sorted_vals = sort(img_double(:), 'descend');
        threshold_val = sorted_vals(round(0.35 * length(sorted_vals)));
        brain_mask = img_double > threshold_val * 0.7;
        brain_mask = bwareaopen(brain_mask, 2000);
        brain_mask = imfill(brain_mask, 'holes');
    end
    
    % Smooth but preserve shape
    brain_mask = imclose(brain_mask, strel('disk', 3));
    brain_mask = imfill(brain_mask, 'holes');
    
    fprintf('  Brain mask: %d pixels (%.1f%% of image)\n', ...
        sum(brain_mask(:)), 100*sum(brain_mask(:))/numel(brain_mask));
end

% FUNCTION: Create 12 Regions with COMPLETE BRAIN COVERAGE
% This function creates 12 regions that TOGETHER cover the ENTIRE brain
% with no gaps and smooth blending at boundaries
function [region_masks, region_centers] = create_12_regions_full_coverage(brain_mask)
    
    [rows, cols] = find(brain_mask);
    if isempty(rows)
        region_masks = {};
        region_centers = [];
        return;
    end
    
    min_row = min(rows); max_row = max(rows);
    min_col = min(cols); max_col = max(cols);
    
    brain_height = max_row - min_row + 1;
    brain_width = max_col - min_col + 1;
    
    % Fixed 3x4 grid for 12 regions
    grid_rows = 3;
    grid_cols = 4;
    
    % Calculate base region size (with overlap to ensure coverage)
    base_height = ceil(brain_height / grid_rows);
    base_width = ceil(brain_width / grid_cols);
    
    % Add significant overlap to ensure complete coverage
    overlap_v = 15;  % Vertical overlap
    overlap_h = 15;  % Horizontal overlap
    
    region_masks = {};
    region_centers = zeros(12, 2);
    region_idx = 0;
    
    fprintf('  Creating 12 regions with full brain coverage...\n');
    fprintf('  Grid: %d rows x %d cols, Base size: %dx%d, Overlap: %dpx\n', ...
        grid_rows, grid_cols, base_height, base_width, max(overlap_v, overlap_h));
    
    for r = 1:grid_rows
        for c = 1:grid_cols
            region_idx = region_idx + 1;
            
            % Calculate region boundaries with overlap
            row_start = min_row + (r-1) * base_height;
            row_end = min(max_row, row_start + base_height - 1 + 2*overlap_v);
            
            col_start = min_col + (c-1) * base_width;
            col_end = min(max_col, col_start + base_width - 1 + 2*overlap_h);
            
            % Adjust start positions to ensure coverage
            if r > 1
                row_start = max(min_row, row_start - overlap_v);
            end
            if c > 1
                col_start = max(min_col, col_start - overlap_h);
            end
            
            % Ensure last region reaches the edge
            if r == grid_rows
                row_end = max_row;
            end
            if c == grid_cols
                col_end = max_col;
            end
            
            % Create region mask
            region_mask = false(size(brain_mask));
            region_mask(row_start:row_end, col_start:col_end) = true;
            region_mask = region_mask & brain_mask;
            
            % Store region
            region_masks{region_idx} = region_mask;
            
            % Calculate center (for blending)
            [r_center, c_center] = find(region_mask);
            if ~isempty(r_center)
                region_centers(region_idx, :) = [mean(r_center), mean(c_center)];
            else
                region_centers(region_idx, :) = [(row_start+row_end)/2, (col_start+col_end)/2];
            end
            
            % Print region info
            fprintf('    Region %2d: rows %3d-%3d, cols %3d-%3d, size: %d pixels\n', ...
                region_idx, row_start, row_end, col_start, col_end, sum(region_mask(:)));
        end
    end
    
    % Verify coverage
    coverage_mask = false(size(brain_mask));
    for i = 1:length(region_masks)
        coverage_mask = coverage_mask | region_masks{i};
    end
    
    coverage_percent = 100 * sum(coverage_mask(:) & brain_mask(:)) / sum(brain_mask(:));
    fprintf('  Total brain coverage: %.1f%%\n', coverage_percent);
    
    if coverage_percent < 99
        fprintf('  Warning: Coverage < 99%%. Adding fill regions...\n');
        
        % Find uncovered areas
        uncovered = brain_mask & ~coverage_mask;
        if any(uncovered(:))
            CC_uncovered = bwconncomp(uncovered);
            for i = 1:min(CC_uncovered.NumObjects, 3)  % Add up to 3 fill regions
                [r_u, c_u] = ind2sub(size(brain_mask), CC_uncovered.PixelIdxList{i});
                if length(r_u) > 50
                    region_idx = region_idx + 1;
                    fill_mask = false(size(brain_mask));
                    fill_mask(CC_uncovered.PixelIdxList{i}) = true;
                    region_masks{region_idx} = fill_mask;
                    region_centers(region_idx, :) = [mean(r_u), mean(c_u)];
                    fprintf('    Added fill region %d: %d pixels\n', region_idx, length(r_u));
                end
            end
        end
    end
    
    fprintf('  Final region count: %d\n', length(region_masks));
end

% FUNCTION: Create Blending Weights for 12 Regions
function blending_weights = create_blending_weights_12regions(region_masks, region_centers, sigma)
    if nargin < 3
        sigma = 25;  % Larger sigma for smoother blending
    end
    
    num_regions = length(region_masks);
    if num_regions == 0
        blending_weights = [];
        return;
    end
    
    [rows, cols] = size(region_masks{1});
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    blending_weights = zeros(rows, cols, num_regions);
    
    for r = 1:num_regions
        % Distance-based weights from region center
        if r <= size(region_centers, 1)
            center_r = region_centers(r, 1);
            center_c = region_centers(r, 2);
            
            % Gaussian weight based on distance to center
            weights = exp(-((X - center_c).^2 + (Y - center_r).^2) / (2 * sigma^2));
            
            % Modulate by region mask
            weights = weights .* region_masks{r};
            
            blending_weights(:,:,r) = weights;
        end
    end
    
    % Normalize so weights sum to 1 at each pixel
    weight_sum = sum(blending_weights, 3);
    weight_sum(weight_sum == 0) = 1;
    
    for r = 1:num_regions
        blending_weights(:,:,r) = blending_weights(:,:,r) ./ weight_sum;
    end
    
    % Apply Gaussian smoothing to weights for even smoother transitions
    for r = 1:num_regions
        blending_weights(:,:,r) = imgaussfilt(blending_weights(:,:,r), 2);
    end
    
    fprintf('  Created blending weights with sigma = %d\n', sigma);
end

% FUNCTION: Wavelet Denoising
function denoised_img = wavelet_denoise_dwi(img, threshold_factor)
    if nargin < 2
        threshold_factor = 2.0;
    end
    
    img_double = im2double(img);
    [coeffs, sizes] = wavedec2(img_double, 2, 'db4');
    
    detail_coeffs = coeffs(end-3*sizes(2,1)*sizes(2,2)+1:end);
    sigma = median(abs(detail_coeffs)) / 0.6745;
    threshold = threshold_factor * sigma;
    
    denoised_coeffs = coeffs;
    cA_length = prod(sizes(1, :));
    start_idx = cA_length + 1;
    
    for level = 1:2
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

% FUNCTION: Frangi Filter for Arteries
function vesselness = frangi_filter_arteries(img, brain_mask, params)
    scales = params.frangi_scales;
    Beta = params.frangi_beta;
    Gamma = params.frangi_gamma;
    
    if ~isa(img, 'double')
        img = im2double(img);
    end
    
    % NO INVERSION - arteries are bright
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
                        if abs(lambda1) < abs(lambda2) * 2
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
    vesselness(~brain_mask) = 0;
end

% FUNCTION: Analyze with 12 Regions and Blending
function vesselness_map = analyze_with_12regions(img, brain_mask, params)
    
    % Create 12 regions with full coverage
    [region_masks, region_centers] = create_12_regions_full_coverage(brain_mask);
    
    if isempty(region_masks)
        vesselness_map = zeros(size(img));
        return;
    end
    
    % Create blending weights
    blending_weights = create_blending_weights_12regions(region_masks, region_centers, 25);
    
    num_regions = length(region_masks);
    region_results = zeros([size(img), num_regions]);
    
    % Analyze each region
    for r = 1:num_regions
        fprintf('    Analyzing region %d/%d...\n', r, num_regions);
        
        region_mask = region_masks{r};
        
        % Extract region
        region_img = img;
        region_img(~region_mask) = 0;
        
        % Enhance region
        region_enhanced = adapthisteq(region_img, ...
            'NumTiles', [4 4], ...
            'ClipLimit', params.cliplimit, ...
            'Distribution', 'rayleigh');
        
        % Apply Frangi
        region_frangi = frangi_filter_arteries(region_enhanced, region_mask, params);
        
        % Adaptive threshold
        local_vals = region_frangi(region_mask);
        if ~isempty(local_vals) && sum(local_vals > 0) > 5
            local_thresh = graythresh(local_vals(local_vals > 0)) * params.vessel_threshold_factor;
            region_frangi = region_frangi .* (region_frangi > local_thresh);
        end
        
        region_results(:,:,r) = region_frangi .* region_mask;
    end
    
    % Blend results using weights
    vesselness_map = zeros(size(img));
    for r = 1:num_regions
        vesselness_map = vesselness_map + region_results(:,:,r) .* blending_weights(:,:,r);
    end
    
    % Final smoothing
    vesselness_map = imgaussfilt(vesselness_map, 1.0);
    vesselness_map = mat2gray(vesselness_map);
    vesselness_map(~brain_mask) = 0;
end

% FUNCTION: Extract Arteries
function [artery_mask, vesselness_map] = extract_arteries(img, brain_mask, params)
    fprintf('  Extracting arteries using 12-region analysis...\n');
    
    vesselness_map = analyze_with_12regions(img, brain_mask, params);
    
    % Erode slightly to remove edge artifacts
    se = strel('disk', params.edge_erosion);
    interior_mask = imerode(brain_mask, se);
    
    % Threshold on interior only
    interior_vals = vesselness_map(interior_mask);
    
    if ~isempty(interior_vals)
        % Adaptive threshold
        opt_thresh = graythresh(interior_vals) * 1.1;
        
        % Get candidates
        candidates = vesselness_map > opt_thresh;
        
        % Keep only interior
        candidates = candidates & interior_mask;
        
        % Clean up
        candidates = bwareaopen(candidates, params.min_vessel_size);
        candidates = imopen(candidates, strel('disk', 1));
        
        artery_mask = candidates;
        
        fprintf('  Detected %d artery pixels (%.2f%% of interior)\n', ...
            sum(artery_mask(:)), 100*sum(artery_mask(:))/sum(interior_mask(:)));
    else
        artery_mask = false(size(vesselness_map));
    end
end

% PROCESSING PIPELINE

fprintf('\nStep 1: Skull stripping...\n');
brain_mask = skull_strip_dwi(original);

fprintf('\nStep 2: Denoising...\n');
brain_denoised = wavelet_denoise_dwi(original, params.wavelet_threshold);
brain_denoised = brain_denoised .* brain_mask;

fprintf('\nStep 3: Enhancement...\n');
brain_enhanced = adapthisteq(brain_denoised, ...
    'NumTiles', [8 8], ...
    'ClipLimit', params.cliplimit, ...
    'Distribution', 'rayleigh');
brain_enhanced = brain_enhanced .* brain_mask;

fprintf('\nStep 4: Artery extraction with 12 regions...\n');
[artery_mask, vesselness_map] = extract_arteries(brain_enhanced, brain_mask, params);

% CREATE VISUALIZATIONS

% Original
orig_disp = original;

% Get 12 regions for display
[region_masks_disp, ~] = create_12_regions_full_coverage(brain_mask);

% Region display
region_disp = repmat(original, [1,1,3]);
colors = lines(12);
for r = 1:min(length(region_masks_disp), 12)
    boundary = bwperim(region_masks_disp{r});
    for c = 1:3
        channel = region_disp(:,:,c);
        channel(boundary) = colors(r,c);
        region_disp(:,:,c) = channel;
    end
end

% Brain display
brain_disp = repmat(original, [1,1,3]);
for c = 1:3
    channel = brain_disp(:,:,c);
    channel(~brain_mask) = 1;
    brain_disp(:,:,c) = channel;
end

% Artery display (red on white)
artery_disp = 255 * ones(size(original), 'uint8');
artery_disp = repmat(artery_disp, [1,1,3]);
artery_disp = im2double(artery_disp);
for c = 1:3
    channel = artery_disp(:,:,c);
    if c == 1
        channel(artery_mask) = 1;
    else
        channel(artery_mask) = 0;
    end
    channel(~brain_mask) = 1;
    artery_disp(:,:,c) = channel;
end

% Coverage verification
coverage_mask = false(size(brain_mask));
for r = 1:length(region_masks_disp)
    coverage_mask = coverage_mask | region_masks_disp{r};
end
coverage_gap = brain_mask & ~coverage_mask;

% Coverage display
coverage_disp = repmat(original, [1,1,3]);
for c = 1:3
    channel = coverage_disp(:,:,c);
    channel(coverage_gap) = 1;  % White for gaps
    if c == 3  % Blue for covered
        channel(coverage_mask) = 0.5;
    else
        channel(coverage_mask) = 0;
    end
    coverage_disp(:,:,c) = channel;
end

% MAIN FIGURE - 4 PANELS
figure('Name', sprintf('DWI Arteries - 12 Regions (Slice %d)', random_idx), ...
       'Position', [100, 100, 1600, 1000]);

% Panel 1: Original
subplot(2,3,1);
imshow(orig_disp);
title('A. Original DWI', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

% Panel 2: 12 Regions
subplot(2,3,2);
imshow(region_disp);
title('B. 12 Analysis Regions (Full Coverage)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

% Panel 3: Coverage Verification
subplot(2,3,3);
imshow(coverage_disp);
title('C. Coverage Check (Blue=Covered, White=Gaps)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;
coverage_pct = 100 * sum(coverage_mask(:)) / sum(brain_mask(:));
text(0.02, 0.95, sprintf('Coverage: %.1f%%', coverage_pct), ...
     'Units', 'normalized', 'Color', 'yellow', 'FontSize', 12, ...
     'BackgroundColor', 'black', 'FontWeight', 'bold');

% Panel 4: Extracted Brain
subplot(2,3,4);
imshow(brain_disp);
title('D. Extracted Brain', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

% Panel 5: Vesselness Map
subplot(2,3,5);
imshow(vesselness_map, []);
title('E. Vesselness Map', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;
colormap(gca, 'hot');
colorbar;

% Panel 6: Final Arteries
subplot(2,3,6);
imshow(artery_disp);
title('F. Extracted Arteries', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

artery_density = 100 * sum(artery_mask(:)) / sum(brain_mask(:));
text(0.02, 0.95, sprintf('Artery Density: %.2f%%', artery_density), ...
     'Units', 'normalized', 'Color', 'blue', 'FontSize', 12, ...
     'BackgroundColor', 'white', 'FontWeight', 'bold');
text(0.02, 0.90, sprintf('Artery Pixels: %d', sum(artery_mask(:))), ...
     'Units', 'normalized', 'Color', 'blue', 'FontSize', 12, ...
     'BackgroundColor', 'white');

sgtitle(sprintf('DWI Artery Extraction with 12 Regions - Slice %d/%d: %s', ...
    random_idx, length(files), files(random_idx).name), 'FontSize', 16, 'FontWeight', 'bold');

% VERIFICATION FIGURE
figure('Name', 'Artery Overlay Verification', 'Position', [150, 150, 900, 900]);

overlay = repmat(original, [1,1,3]);
artery_outline = bwperim(artery_mask);
for c = 1:3
    channel = overlay(:,:,c);
    if c == 1
        channel(artery_outline) = 1;
    else
        channel(artery_outline) = 0;
    end
    overlay(:,:,c) = channel;
end
imshow(overlay);
title('Artery Outlines (Red) on Original DWI', 'FontSize', 16, 'FontWeight', 'bold');
axis image off;

% STATISTICS
fprintf('\n%s\n', repmat('-', 1, 70));
fprintf('ARTERY EXTRACTION RESULTS - 12 REGIONS (FULL COVERAGE)\n');
fprintf('%s\n', repmat('-', 1, 70));

fprintf('\nImage: %s\n', files(random_idx).name);
fprintf('Brain pixels: %d (%.1f%% of image)\n', ...
    sum(brain_mask(:)), 100*sum(brain_mask(:))/numel(brain_mask));

% Region coverage
coverage_final = false(size(brain_mask));
for r = 1:length(region_masks_disp)
    coverage_final = coverage_final | region_masks_disp{r};
end
final_coverage = 100 * sum(coverage_final(:) & brain_mask(:)) / sum(brain_mask(:));
fprintf('Brain coverage by 12 regions: %.1f%%\n', final_coverage);

fprintf('\nARTERY STATISTICS:\n');
fprintf('  Artery pixels detected: %d\n', sum(artery_mask(:)));
fprintf('  Artery density: %.2f%% of brain\n', 100*sum(artery_mask(:))/sum(brain_mask(:)));

% Check for edge artifacts
se = strel('disk', params.edge_erosion);
interior_check = imerode(brain_mask, se);
edge_arteries = sum(artery_mask(:) & ~interior_check(:));
if edge_arteries == 0
    fprintf('  No arteries at brain edge\n');
else
    fprintf('  %d arteries at edge (%.1f%%)\n', ...
        edge_arteries, 100*edge_arteries/sum(artery_mask(:)));
end

fprintf('\nREGION DETAILS:\n');
fprintf('  Number of regions: %d\n', length(region_masks_disp));
fprintf('  Grid: 3 rows x 4 columns\n');
fprintf('  Overlap: 15 pixels\n');
fprintf('  Blending sigma: 25\n');

fprintf('\n%s\n', repmat('-', 1, 70));
fprintf('DWI Artery Extraction Complete!\n');