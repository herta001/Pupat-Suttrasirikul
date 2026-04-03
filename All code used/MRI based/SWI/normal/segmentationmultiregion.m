clear all; clc; close all;

% LOAD IMAGE
% Look for common image formats
files = [dir('*.jpg'); dir('*.png'); dir('*.tif'); dir('*.bmp'); dir('*.dcm')];
if isempty(files)
    error('No image files found. Place an SWI image in current folder.');
end

% Load random image
img = imread(files(randi(length(files))).name);

% Convert to grayscale if needed
if size(img, 3) == 3
    original = rgb2gray(img);
else
    original = img;
end

% FUNCTION: Skull Stripping
function brain_mask = skull_strip_swi(mri_image)
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
    
    % Remove small protrusions (potential skull remnants)
    brain_mask = bwmorph(brain_mask, 'spur', 5);
    
    % Final smoothing of brain boundary
    brain_mask = imdilate(brain_mask, strel('disk', 2));
end

% FUNCTION: Create Brain Interior Mask
function interior_mask = create_brain_interior_mask(brain_mask, erosion_pixels)
    if nargin < 2
        erosion_pixels = 8;  % Erode by 8 pixels to remove edge
    end
    
    % Erode brain mask to get interior (away from edge)
    se = strel('disk', erosion_pixels);
    interior_mask = imerode(brain_mask, se);
    
    % Also create a boundary region to suppress
    boundary_mask = brain_mask & ~interior_mask;
    
    fprintf('  Brain interior: %d pixels (%.1f%% of brain)\n', ...
        sum(interior_mask(:)), 100*sum(interior_mask(:))/sum(brain_mask(:)));
    fprintf('  Brain boundary (suppressed): %d pixels\n', sum(boundary_mask(:)));
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

% FUNCTION: Texture Feature Extraction
function texture_map = extract_vessel_texture(img, brain_mask, interior_mask)
    img_double = im2double(img);
    texture_map = zeros(size(img_double));
    
    window_size = 7;
    padded_img = padarray(img_double, [floor(window_size/2), floor(window_size/2)], 'symmetric');
    
    [rows, cols] = size(img_double);
    margin = floor(window_size/2);
    
    for i = margin+1:rows-margin
        for j = margin+1:cols-margin
            % Only process interior of brain (not edges)
            if interior_mask(i, j)
                window = padded_img(i:i+window_size-1, j:j+window_size-1);
                
                window_norm = window / sum(window(:) + eps);
                entropy_val = -sum(window_norm(:) .* log2(window_norm(:) + eps));
                std_val = std(window(:));
                range_val = max(window(:)) - min(window(:));
                
                [gx, gy] = gradient(window);
                grad_mag = mean(sqrt(gx(:).^2 + gy(:).^2));
                
                energy_val = sum(window(:).^2);
                non_uniformity = 1 - energy_val;
                
                texture_map(i, j) = 0.25 * entropy_val + ...
                                   0.25 * std_val + ...
                                   0.15 * range_val + ...
                                   0.20 * grad_mag + ...
                                   0.15 * non_uniformity;
            end
        end
    end
    
    if max(texture_map(:)) > 0
        texture_map = texture_map / max(texture_map(:));
    end
    texture_map = imgaussfilt(texture_map, 1.0);
end

% FUNCTION: Local Binary Patterns (LBP) for Texture
function lbp_map = extract_lbp_texture(img, brain_mask, interior_mask)
    img_double = im2double(img);
    lbp_map = zeros(size(img_double));
    
    radius = 2;
    neighbors = 8;
    
    offsets = zeros(neighbors, 2);
    for n = 1:neighbors
        angle = 2 * pi * (n-1) / neighbors;
        offsets(n, 1) = round(radius * cos(angle));
        offsets(n, 2) = round(radius * sin(angle));
    end
    
    pad_size = radius + 1;
    padded_img = padarray(img_double, [pad_size, pad_size], 'symmetric');
    
    [rows, cols] = size(img_double);
    
    for i = 1:rows
        for j = 1:cols
            if interior_mask(i, j)  % Only interior
                center = padded_img(i+pad_size, j+pad_size);
                lbp_code = 0;
                
                for n = 1:neighbors
                    ni = i + pad_size + offsets(n, 1);
                    nj = j + pad_size + offsets(n, 2);
                    neighbor_val = padded_img(ni, nj);
                    
                    if neighbor_val >= center
                        lbp_code = lbp_code + 2^(n-1);
                    end
                end
                
                binary = dec2bin(lbp_code, neighbors);
                transitions = 0;
                for n = 1:neighbors-1
                    if binary(n) ~= binary(n+1)
                        transitions = transitions + 1;
                    end
                end
                if binary(end) ~= binary(1)
                    transitions = transitions + 1;
                end
                
                if transitions <= 2
                    num_ones = sum(binary == '1');
                    lbp_map(i, j) = num_ones / neighbors;
                else
                    lbp_map(i, j) = 0.3;
                end
            end
        end
    end
    
    if max(lbp_map(:)) > 0
        lbp_map = lbp_map / max(lbp_map(:));
    end
end

% FUNCTION: Frangi Vesselness Filter
function vesselness = frangi_vesselness_swi(img, interior_mask)
    scales = [1.5, 2, 2.5, 3, 3.5, 4];
    Alpha = 0.52; Beta = 0.52; Gamma = 0.26;
    
    if ~isa(img, 'double')
        img = im2double(img);
    end
    img_processed = 1 - img;  % Keep inversion for dark vessels
    
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
        
        A = 2 * Alpha^2;
        B = 2 * Beta^2;
        C = 2 * Gamma^2;
        
        response = exp(-Rb.^2 ./ B) .* (1 - exp(-S.^2 ./ C));
        response(lambda2 >= -0.02) = 0;
        
        vesselness = max(vesselness, response);
    end
    
    % Apply edge suppression - reduce response at brain boundary
    edge_suppression = ones(size(img));
    edge_suppression(~interior_mask) = 0.1;  % 90% suppression at edges
    vesselness = vesselness .* edge_suppression;
    
    vesselness = imgaussfilt(vesselness, 0.5);
    vesselness = mat2gray(vesselness);
    
    % Final mask to interior only
    vesselness(~interior_mask) = 0;
end

% FUNCTION: Split Brain into Smaller Non-Overlapping Regions (12 Regions)
function [regions, region_masks, region_indices] = split_into_focused_regions(interior_mask, num_regions)
    % Split the brain interior into smaller non-overlapping regions
    
    % Get bounding box of interior
    [rows, cols] = find(interior_mask);
    if isempty(rows)
        error('No interior pixels found');
    end
    
    min_row = min(rows); max_row = max(rows);
    min_col = min(cols); max_col = max(cols);
    
    brain_height = max_row - min_row + 1;
    brain_width = max_col - min_col + 1;
    
    % Calculate grid dimensions
    grid_rows = 3;
    grid_cols = 4;
    
    % Calculate region sizes
    region_height = floor(brain_height / grid_rows);
    region_width = floor(brain_width / grid_cols);
    
    % Ensure regions are at least 40 pixels
    region_height = max(region_height, 40);
    region_width = max(region_width, 40);
    
    % Initialize output
    regions = {};
    region_masks = {};
    region_indices = {};
    
    region_count = 0;
    
    % Create non-overlapping grid regions with overlap at boundaries
    overlap = 10;  % Small overlap to prevent seam artifacts
    
    for r = 1:grid_rows
        for c = 1:grid_cols
            if region_count >= num_regions
                break;
            end
            
            % Calculate region boundaries with overlap
            row_start = max(min_row, min_row + (r-1) * region_height - overlap);
            row_end = min(max_row, row_start + region_height - 1 + 2*overlap);
            
            col_start = max(min_col, min_col + (c-1) * region_width - overlap);
            col_end = min(max_col, col_start + region_width - 1 + 2*overlap);
            
            % Ensure we don't exceed bounds
            row_start = max(min_row, row_start);
            row_end = min(max_row, row_end);
            col_start = max(min_col, col_start);
            col_end = min(max_col, col_end);
            
            % Create region mask
            region_mask = false(size(interior_mask));
            region_mask(row_start:row_end, col_start:col_end) = true;
            region_mask = region_mask & interior_mask;
            
            % Store region if it has enough brain tissue
            if sum(region_mask(:)) > 200
                region_count = region_count + 1;
                regions{region_count} = [row_start, row_end, col_start, col_end];
                region_masks{region_count} = region_mask;
                region_indices{region_count} = find(region_mask);
            end
        end
        if region_count >= num_regions
            break;
        end
    end
    
    % Fill any gaps
    if region_count < num_regions
        combined_mask = false(size(interior_mask));
        for i = 1:region_count
            combined_mask = combined_mask | region_masks{i};
        end
        
        uncovered = interior_mask & ~combined_mask;
        if any(uncovered(:))
            CC = bwconncomp(uncovered);
            for i = 1:min(CC.NumObjects, num_regions - region_count)
                region_count = region_count + 1;
                region_mask = false(size(interior_mask));
                region_mask(CC.PixelIdxList{i}) = true;
                region_masks{region_count} = region_mask;
                region_indices{region_count} = CC.PixelIdxList{i};
                
                [r_sub, c_sub] = ind2sub(size(interior_mask), CC.PixelIdxList{i});
                regions{region_count} = [min(r_sub), max(r_sub), min(c_sub), max(c_sub)];
            end
        end
    end
    
    fprintf('  Created %d regions with %d-pixel overlap\n', length(regions), overlap);
end

% FUNCTION: Create Blending Weights
function blending_weights = create_blending_weights(region_masks, sigma)
    if nargin < 2
        sigma = 20;  % Larger sigma for smoother transitions
    end
    
    num_regions = length(region_masks);
    [rows, cols] = size(region_masks{1});
    
    % Initialize distance maps
    distance_maps = zeros(rows, cols, num_regions);
    
    for r = 1:num_regions
        % Compute distance from region boundary
        dist = bwdist(~region_masks{r});
        dist = dist / max(dist(:) + eps);
        
        % Convert to Gaussian-like weights
        distance_maps(:,:,r) = exp(-(1 - dist).^2 / (2 * 0.3^2));
        distance_maps(:,:,r) = distance_maps(:,:,r) .* region_masks{r};
    end
    
    % Normalize
    weight_sum = sum(distance_maps, 3);
    weight_sum(weight_sum == 0) = 1;
    
    blending_weights = zeros(rows, cols, num_regions);
    for r = 1:num_regions
        blending_weights(:,:,r) = distance_maps(:,:,r) ./ weight_sum;
    end
    
    % Smooth weights
    for r = 1:num_regions
        blending_weights(:,:,r) = imgaussfilt(blending_weights(:,:,r), 2);
    end
end

% FUNCTION: Analyze Each Region with Blending
function blended_vesselness = analyze_regions_with_blending(img, interior_mask, num_regions)
    % Split brain interior into regions
    [region_coords, region_masks, ~] = split_into_focused_regions(interior_mask, num_regions);
    num_valid_regions = length(region_masks);
    
    % Create blending weights
    fprintf('    Creating blending weights...\n');
    blending_weights = create_blending_weights(region_masks, 20);
    
    % Initialize vesselness results
    region_results = zeros([size(img), num_valid_regions]);
    
    % Analyze each region
    for r = 1:num_valid_regions
        fprintf('    Analyzing region %d/%d...\n', r, num_valid_regions);
        
        region_mask = region_masks{r};
        
        % Create region-specific image
        region_img = img;
        region_img(~region_mask) = 0;
        
        % Apply local contrast enhancement
        region_enhanced = adapthisteq(region_img, ...
            'NumTiles', [4 4], ...
            'ClipLimit', 0.015, ...
            'Distribution', 'rayleigh', ...
            'Alpha', 0.4);
        
        % Apply Frangi with edge suppression
        region_double = im2double(region_enhanced);
        region_frangi = frangi_vesselness_swi(region_double, region_mask);
        
        % Extract texture features (only within region)
        region_texture = extract_vessel_texture(region_enhanced, region_mask, region_mask);
        region_lbp = extract_lbp_texture(region_enhanced, region_mask, region_mask);
        
        % Combine features
        texture_boost = 0.25 * region_texture + 0.15 * region_lbp;
        region_result = region_frangi .* (1 + texture_boost);
        
        % Adaptive thresholding
        local_values = region_result(region_mask);
        if ~isempty(local_values) && sum(local_values > 0) > 10
            local_threshold = graythresh(local_values(local_values > 0)) * 0.85;
            region_result = region_result .* (region_result > local_threshold);
        end
        
        % Store result
        region_results(:,:,r) = region_result .* region_mask;
    end
    
    % Blend results
    fprintf('    Blending region results...\n');
    blended_vesselness = zeros(size(img));
    
    for r = 1:num_valid_regions
        blended_vesselness = blended_vesselness + region_results(:,:,r) .* blending_weights(:,:,r);
    end
    
    % Apply smoothing and normalization
    blended_vesselness = imgaussfilt(blended_vesselness, 1.0);
    
    if max(blended_vesselness(:)) > 0
        blended_vesselness = blended_vesselness / max(blended_vesselness(:));
    end
    blended_vesselness(~interior_mask) = 0;
end

% FUNCTION: Main Vessel Extraction
function [vessel_mask, vesselness_map] = extract_brain_vessels(img, brain_mask, interior_mask)
    fprintf('  Performing vessel extraction on brain interior...\n');
    
    % Analyze with 12 regions
    vesselness_map = analyze_regions_with_blending(img, interior_mask, 12);
    
    % Normalize
    vesselness_map = mat2gray(vesselness_map);
    vesselness_map(~interior_mask) = 0;
    
    % Adaptive thresholding
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
        
        % Remove any remaining boundary artifacts
        boundary_dilate = imdilate(~interior_mask, strel('disk', 3));
        vessel_mask = vessel_mask & ~boundary_dilate;
    else
        vessel_mask = false(size(vesselness_map));
    end
    
    % Final smoothing
    vesselness_map = imgaussfilt(vesselness_map, 0.5);
end

% PROCESSING PIPELINE
fprintf('\n%s\n', repmat('-', 1, 70));
fprintf('SWI VESSEL EXTRACTION\n');
fprintf('%s\n', repmat('-', 1, 70));

% Step 1: Skull stripping
fprintf('\nStep 1: Removing skull...\n');
brain_mask = skull_strip_swi(original);
brain_only = original;
brain_only(~brain_mask) = 0;

% Step 2: Create brain interior mask (remove edge)
fprintf('Step 2: Creating brain interior mask (removing edge)...\n');
interior_mask = create_brain_interior_mask(brain_mask, 8);

% Step 3: Wavelet denoising
fprintf('Step 3: Applying wavelet denoising...\n');
brain_denoised = wavelet_denoise_swi(brain_only);
brain_denoised(~brain_mask) = 0;

% Step 4: Global CLAHE enhancement
fprintf('Step 4: Applying global contrast enhancement...\n');
brain_enhanced = adapthisteq(brain_denoised, ...
    'NumTiles', [8 8], ...
    'ClipLimit', 0.02, ...
    'Distribution', 'rayleigh', ...
    'Alpha', 0.5);
brain_enhanced(~brain_mask) = 0;

% Step 5: VESSEL EXTRACTION (INTERIOR ONLY)
fprintf('Step 5: Performing vessel extraction (interior only)...\n');
[vessel_mask, vesselness_map] = extract_brain_vessels(brain_enhanced, brain_mask, interior_mask);

% CREATE THE FOUR KEY VISUALIZATIONS

% 1. ORIGINAL IMAGE
original_display = original;

% 2. SPLIT BRAIN REGIONS (with grid overlay)
regions_display = original;
regions_display = im2double(regions_display);
regions_display = repmat(regions_display, [1,1,3]); % Convert to RGB

% Get region boundaries for display
[region_coords, region_masks, ~] = split_into_focused_regions(interior_mask, 12);

% Draw region boundaries in different colors
colors = lines(12);
for r = 1:length(region_coords)
    coords = region_coords{r};
    % Draw rectangle
    for x = coords(3):coords(4)
        if x >= 1 && x <= size(regions_display,2)
            if coords(1) >= 1 && coords(1) <= size(regions_display,1)
                regions_display(coords(1), x, :) = reshape(colors(r,:), [1,1,3]);
            end
            if coords(2) >= 1 && coords(2) <= size(regions_display,1)
                regions_display(coords(2), x, :) = reshape(colors(r,:), [1,1,3]);
            end
        end
    end
    for y = coords(1):coords(2)
        if y >= 1 && y <= size(regions_display,1)
            if coords(3) >= 1 && coords(3) <= size(regions_display,2)
                regions_display(y, coords(3), :) = reshape(colors(r,:), [1,1,3]);
            end
            if coords(4) >= 1 && coords(4) <= size(regions_display,2)
                regions_display(y, coords(4), :) = reshape(colors(r,:), [1,1,3]);
            end
        end
    end
end

% 3. EXTRACTED BRAIN (brain only, no background)
brain_display = brain_only;
brain_display = im2double(brain_display);
brain_display = repmat(brain_display, [1,1,3]); % Convert to RGB
% Set background to white
for c = 1:3
    channel = brain_display(:,:,c);
    channel(~brain_mask) = 1;
    brain_display(:,:,c) = channel;
end

% 4. RESULT - SEGMENTED BLOOD VESSELS (black on white)
result_display = 255 * ones(size(brain_enhanced), 'uint8');
result_display = repmat(result_display, [1,1,3]); % RGB
result_display = im2double(result_display);
% Set vessels to black
for c = 1:3
    channel = result_display(:,:,c);
    channel(vessel_mask) = 0;
    result_display(:,:,c) = channel;
end
% Set background to white
for c = 1:3
    channel = result_display(:,:,c);
    channel(~brain_mask) = 1;
    result_display(:,:,c) = channel;
end

% MAIN VISUALIZATION - 4-PANEL PLOT
figure('Name', 'SWI Vessel Extraction Pipeline', ...
       'Position', [100, 100, 1400, 1000], ...
       'Color', 'white');

% Panel 1: Original Image
subplot(2,2,1);
imshow(original_display);
title('A. ORIGINAL SWI IMAGE', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;
% Add annotation
text(0.02, 0.98, 'Raw input image', 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, ...
     'BackgroundColor', 'black');

% Panel 2: Split Brain Regions
subplot(2,2,2);
imshow(regions_display);
title('B. SPLIT BRAIN REGIONS (12 Non-Overlapping)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;
text(0.02, 0.98, '12 analysis regions with overlap', 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, ...
     'BackgroundColor', 'black');

% Panel 3: Extracted Brain
subplot(2,2,3);
imshow(brain_display);
title('C. EXTRACTED BRAIN (Skull Stripped)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;
text(0.02, 0.98, 'Brain tissue only, edge suppressed', 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, ...
     'BackgroundColor', 'black');

% Panel 4: Result - Segmented Blood Vessels
subplot(2,2,4);
imshow(result_display);
title('D. SEGMENTED BLOOD VESSELS', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;
text(0.02, 0.98, 'Final vessel segmentation', 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'Color', 'red', 'FontSize', 10, ...
     'BackgroundColor', 'white');
% Add vessel density info
vessel_density = 100 * sum(vessel_mask(:)) / sum(interior_mask(:));
text(0.02, 0.92, sprintf('Vessel Density: %.2f%%', vessel_density), ...
     'Units', 'normalized', 'VerticalAlignment', 'top', 'Color', 'blue', ...
     'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'white');

% Add main title
sgtitle('SWI Brain Vessel Extraction Pipeline', 'FontSize', 16, 'FontWeight', 'bold');

% SUPPLEMENTARY FIGURE - Detailed View
figure('Name', 'SWI Vessel Extraction - Detailed Comparison', ...
       'Position', [150, 150, 1200, 600], ...
       'Color', 'white');

% Original vs Result side by side
subplot(1,2,1);
imshow(original_display);
title('ORIGINAL SWI', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

subplot(1,2,2);
imshow(result_display);
title('SEGMENTED BLOOD VESSELS', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

% Add overlay of vessel outlines on original for verification
figure('Name', 'Vessel Overlay Verification', ...
       'Position', [200, 200, 800, 800], ...
       'Color', 'white');

verification_img = im2double(original);
verification_img = repmat(verification_img, [1,1,3]);
% Draw vessel outlines in red
vessel_outline = bwperim(vessel_mask);
for c = 1:3
    channel = verification_img(:,:,c);
    if c == 1 % Red channel
        channel(vessel_outline) = 1;
    else
        channel(vessel_outline) = 0;
    end
    verification_img(:,:,c) = channel;
end
imshow(verification_img);
title('Vessel Outlines (Red) Overlaid on Original', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

% STATISTICS
fprintf('\n%s\n', repmat('-', 1, 70));
fprintf('VESSEL EXTRACTION RESULTS - 4-PANEL SUMMARY\n');
fprintf('%s\n', repmat('-', 1, 70));

brain_pixels = sum(brain_mask(:));
interior_pixels = sum(interior_mask(:));
edge_pixels = brain_pixels - interior_pixels;
vessel_pixels = sum(vessel_mask(:));
vessel_density = (vessel_pixels / interior_pixels) * 100;

fprintf('\nBRAIN STATISTICS:\n');
fprintf('  Total brain pixels: %d\n', brain_pixels);
fprintf('  Interior brain pixels: %d (%.1f%%)\n', interior_pixels, 100*interior_pixels/brain_pixels);
fprintf('  Edge pixels suppressed: %d (%.1f%%)\n', edge_pixels, 100*edge_pixels/brain_pixels);

fprintf('\nVESSEL STATISTICS:\n');
fprintf('  Vessel pixels detected: %d\n', vessel_pixels);
fprintf('  Vessel density: %.2f%% of brain interior\n', vessel_density);

fprintf('\nREGION STATISTICS:\n');
fprintf('  Number of analysis regions: 12\n');
fprintf('  Region overlap: 10 pixels\n');
fprintf('  Grid arrangement: 3 rows x 4 columns\n');

fprintf('\nARTIFACT SUPPRESSION:\n');
% Check if any vessels were detected at the edge
edge_vessels = sum(vessel_mask(:) & ~interior_mask(:));
if edge_vessels == 0
    fprintf('  No vessels detected at brain edge\n');
else
    fprintf('  %d vessels still at edge - consider increasing erosion\n', edge_vessels);
end

fprintf('%s\n', repmat('-', 1, 70));
fprintf('\nProcessing Complete!\n');
fprintf('Generated:\n');
fprintf('  1. Figure 1: 4-Panel Pipeline View\n');
fprintf('  2. Figure 2: Original vs Result Comparison\n');
fprintf('  3. Figure 3: Vessel Overlay Verification\n');