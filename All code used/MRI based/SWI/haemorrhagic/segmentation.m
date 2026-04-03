% SWI BRAIN VESSEL EXTRACTION & HEMORRHAGE DETECTION (DARK BLOOD)

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
    
    % Additional cleanup: remove any holes and ensure convexity
    brain_mask = imfill(brain_mask, 'holes');
end

% FUNCTION: Create Brain Interior Mask (STRICTER - removes edge artifacts)
function interior_mask = create_brain_interior_mask(brain_mask, erosion_pixels)
    if nargin < 2
        erosion_pixels = 10;  % Increased from 8 to 10 for stricter edge removal
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
    edge_suppression(~interior_mask) = 0.05;  % Increased suppression to 95% (was 90%)
    vesselness = vesselness .* edge_suppression;
    
    vesselness = imgaussfilt(vesselness, 0.5);
    vesselness = mat2gray(vesselness);
    
    % Final mask to interior only
    vesselness(~interior_mask) = 0;
end

% FUNCTION: Split Brain into Larger Regions (12 Regions - Optimized Size)
function [regions, region_masks, region_indices] = split_into_focused_regions(interior_mask, num_regions)
    % Split the brain interior into 12 larger regions for robust analysis
    
    % Get bounding box of interior
    [rows, cols] = find(interior_mask);
    if isempty(rows)
        error('No interior pixels found');
    end
    
    min_row = min(rows); max_row = max(rows);
    min_col = min(cols); max_col = max(cols);
    
    brain_height = max_row - min_row + 1;
    brain_width = max_col - min_col + 1;
    
    % Calculate grid dimensions for 12 larger regions (3 rows x 4 columns)
    grid_rows = 3;
    grid_cols = 4;
    
    % Calculate region sizes - larger for better statistical power
    region_height = floor(brain_height / grid_rows);
    region_width = floor(brain_width / grid_cols);
    
    % Ensure regions are at least 60 pixels (larger minimum)
    region_height = max(region_height, 60);
    region_width = max(region_width, 60);
    
    % Initialize output
    regions = {};
    region_masks = {};
    region_indices = {};
    
    region_count = 0;
    
    % Create grid regions with moderate overlap
    overlap = 8;  % Moderate overlap for smooth transitions
    
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
            
            % Store region if it has enough brain tissue (higher threshold for larger regions)
            if sum(region_mask(:)) > 300  % Increased from 200 to 300 for larger regions
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
    
    fprintf('  Created %d larger regions with %d-pixel overlap\n', length(regions), overlap);
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
    
    % Analyze with 12 regions (optimized size)
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

% NEW FUNCTION: Vein Detection (to differentiate from hemorrhage)
function vein_mask = detect_veins(img, brain_mask, interior_mask)
    fprintf('    Detecting veins for differentiation...\n');
    
    img_double = im2double(img);
    
    % Create strict brain mask
    strict_brain_mask = brain_mask & interior_mask;
    
    % Use Frangi filter to detect tubular structures
    img_inverted = 1 - img_double;
    
    % Multi-scale vesselness filter
    scales = [1, 1.5, 2, 2.5, 3];
    vesselness = zeros(size(img_double));
    
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
        
        Dxx = conv2(img_inverted, Gxx, 'same');
        Dxy = conv2(img_inverted, Gxy, 'same');
        Dyy = conv2(img_inverted, Gyy, 'same');
        
        % Compute eigenvalues
        lambda1 = zeros(size(img_double));
        lambda2 = zeros(size(img_double));
        
        for i = 1:size(img_double, 1)
            for j = 1:size(img_double, 2)
                H = [Dxx(i,j), Dxy(i,j); Dxy(i,j), Dyy(i,j)];
                eigvals = eig(H);
                lambda1(i,j) = max(eigvals);
                lambda2(i,j) = min(eigvals);
            end
        end
        
        Rb = abs(lambda1) ./ (abs(lambda2) + eps);
        S = sqrt(lambda1.^2 + lambda2.^2);
        
        Beta = 0.5;
        Gamma = 0.25;
        
        response = exp(-Rb.^2 / (2*Beta^2)) .* (1 - exp(-S.^2 / (2*Gamma^2)));
        response(lambda2 >= 0) = 0;
        
        vesselness = max(vesselness, response);
    end
    
    % Normalize
    vesselness = mat2gray(vesselness);
    vesselness(~strict_brain_mask) = 0;
    
    % Threshold for vein detection
    vein_vals = vesselness(strict_brain_mask);
    if ~isempty(vein_vals)
        vein_thresh = graythresh(vein_vals) * 0.7;
        vein_mask = vesselness > vein_thresh;
    else
        vein_mask = false(size(vesselness));
    end
    
    % Keep only elongated structures
    vein_mask = bwareaopen(vein_mask, 10);
    vein_props = regionprops(vein_mask, 'Eccentricity', 'Area', 'Perimeter');
    
    if ~isempty(vein_props)
        final_veins = false(size(vein_mask));
        for k = 1:length(vein_props)
            circularity = 4*pi*vein_props(k).Area / (vein_props(k).Perimeter^2 + eps);
            if (circularity < 0.3 || vein_props(k).Eccentricity > 0.7) && vein_props(k).Area > 15
                final_veins = final_veins | ismember(labelmatrix(bwconncomp(vein_mask)), k);
            end
        end
        vein_mask = final_veins;
    end
    
    % Dilate slightly
    vein_mask = imdilate(vein_mask, strel('disk', 2));
    vein_mask = vein_mask & strict_brain_mask;
    
    fprintf('    Detected %d vein pixels\n', sum(vein_mask(:)));
end

% BEST PERFORMING FUNCTION: Dark Hemorrhage Detection (Slightly Increased Threshold)
function [hemorrhage_mask, region_scores, hemorrhage_regions] = detect_dark_hemorrhage(img, brain_mask, interior_mask, region_masks, vein_mask)
    fprintf('\n  Detecting DARK hemorrhage signs (slightly increased threshold)...\n');
    
    img_double = im2double(img);
    hemorrhage_mask = false(size(img_double));
    num_regions = length(region_masks);
    region_scores = zeros(num_regions, 10);
    hemorrhage_regions = [];
    
    % Create a strict brain mask
    strict_brain_mask = brain_mask & interior_mask;
    
    % Get global brain statistics
    brain_vals = img_double(strict_brain_mask);
    if isempty(brain_vals)
        fprintf('  WARNING: No valid brain pixels found!\n');
        return;
    end
    
    global_mean = mean(brain_vals);
    global_std = std(brain_vals);
    global_median = median(brain_vals);
    
    % SLIGHTLY INCREASED THRESHOLDS
    low_percentile = prctile(brain_vals, 21); % Increased from 22 to 21
    very_low_percentile = prctile(brain_vals, 11); % Increased from 12 to 11
    
    for r = 1:num_regions
        region_mask = region_masks{r};
        region_mask = region_mask & strict_brain_mask;
        region_pixels = img_double(region_mask);
        
        if isempty(region_pixels) || sum(region_mask(:)) < 50
            region_scores(r, :) = 0;
            continue;
        end
        
        mean_intensity = mean(region_pixels);
        median_intensity = median(region_pixels);
        min_intensity = min(region_pixels);
        
        % 1. INTENSITY SCORE - Slightly increased
        intensity_z = (global_mean - mean_intensity) / (global_std + eps);
        intensity_z_median = (global_median - median_intensity) / (global_std + eps);
        intensity_z_min = (global_mean - min_intensity) / (global_std + eps);
        
        intensity_score = sigmoid(max([intensity_z, intensity_z_median, intensity_z_min]), 0.49, 2.5); % Increased from 0.48 to 0.49
        
        % 2. DARK PIXEL PROPORTION - Slightly increased
        dark_pixel_thresh1 = global_mean - 1.29 * global_std; % Increased from 1.28 to 1.29
        dark_pixel_thresh2 = low_percentile;
        dark_pixel_thresh3 = very_low_percentile;
        
        dark_ratio1 = sum(region_pixels < dark_pixel_thresh1) / (numel(region_pixels) + eps);
        dark_ratio2 = sum(region_pixels < dark_pixel_thresh2) / (numel(region_pixels) + eps);
        dark_ratio3 = sum(region_pixels < dark_pixel_thresh3) / (numel(region_pixels) + eps);
        
        dark_score = max([sigmoid(dark_ratio1 * 5, 0.145, 6.0), ... % Increased from 0.14 to 0.145
                         sigmoid(dark_ratio2 * 6, 0.115, 6.0), ... % Increased from 0.11 to 0.115
                         sigmoid(dark_ratio3 * 8, 0.075, 6.0)]); % Increased from 0.07 to 0.075
        
        % 3. LOCAL DARKNESS - Slightly increased
        neighborhood = imdilate(region_mask, strel('disk', 8)) & ~region_mask & strict_brain_mask;
        if any(neighborhood(:))
            neighborhood_mean = mean(img_double(neighborhood));
            neighborhood_std = std(img_double(neighborhood));
            local_darkness = (neighborhood_mean - mean_intensity) / (neighborhood_mean + eps);
            local_z = (neighborhood_mean - mean_intensity) / (neighborhood_std + eps);
            local_dark_score = max(sigmoid(local_darkness * 8, 0.115, 7.0), ... % Increased from 0.11 to 0.115
                                   sigmoid(local_z, 0.89, 2.0)); % Increased from 0.88 to 0.89
        else
            local_dark_score = intensity_score * 0.87; % Increased from 0.88 to 0.87
        end
        
        % 4. Find ALL dark clusters - Slightly increased minimum sizes
        dark_clusters_all = (img_double < dark_pixel_thresh1) & region_mask & strict_brain_mask;
        dark_clusters_all = bwareaopen(dark_clusters_all, 3); % Kept at 3
        
        dark_clusters_main = (img_double < dark_pixel_thresh2) & region_mask & strict_brain_mask;
        dark_clusters_main = bwareaopen(dark_clusters_main, 4); % Kept at 4
        
        dark_clusters_deep = (img_double < dark_pixel_thresh3) & region_mask & strict_brain_mask;
        dark_clusters_deep = bwareaopen(dark_clusters_deep, 3); % Kept at 3
        
        % Combine all dark clusters
        all_dark = dark_clusters_all | dark_clusters_main | dark_clusters_deep;
        
        % 5. SIZE SCORE - Slightly increased thresholds
        size_score = 0;
        very_dark_score = 0;
        cluster_count_score = 0;
        max_area = 0;
        total_dark_area = 0;
        avg_cluster_area = 0;
        
        if any(all_dark(:))
            cluster_props = regionprops(all_dark, 'Area', 'PixelIdxList', 'Eccentricity', 'Solidity');
            areas = [cluster_props.Area];
            
            max_area = max(areas);
            size_score = sigmoid(max_area, 21, 0.05); % Increased from 20 to 21
            
            total_dark_area = sum(areas);
            avg_cluster_area = mean(areas);
            total_area_score = sigmoid(total_dark_area, 72, 0.02); % Increased from 70 to 72
            size_score = max(size_score, total_area_score * 0.71); % Increased from 0.72 to 0.71
            
            if length(areas) > 2
                cluster_count_score = sigmoid(length(areas), 3, 0.6) * 0.21; % Increased from 0.22 to 0.21
            end
            
            if any(dark_clusters_deep(:))
                deep_props = regionprops(dark_clusters_deep, 'Area');
                deep_areas = [deep_props.Area];
                max_deep = max(deep_areas);
                very_dark_score = sigmoid(max_deep, 11.5, 0.07) * 0.31; % Increased from 0.32 to 0.31
            end
        end
        
        % 6. VEIN OVERLAP - Slightly increased penalty
        vein_overlap_ratio = 0;
        if sum(all_dark(:)) > 0
            vein_overlap = sum(all_dark(:) & vein_mask(:));
            total_dark = sum(all_dark(:));
            vein_overlap_ratio = vein_overlap / (total_dark + eps);
        end
        
        % 7. HOMOGENEITY
        std_intensity = std(region_pixels);
        cv = std_intensity / (mean_intensity + eps);
        homogeneity_score = 1 - sigmoid(cv, 0.69, 3.0); % Increased from 0.68 to 0.69
        
        % 8. CONTRAST
        region_dilated = imdilate(region_mask, strel('disk', 5));
        region_boundary = region_dilated & ~region_mask & strict_brain_mask;
        
        if any(region_boundary(:)) && any(region_mask(:))
            boundary_vals = img_double(region_boundary);
            interior_vals = img_double(region_mask);
            mean_contrast = mean(interior_vals) - mean(boundary_vals);
            median_contrast = median(interior_vals) - median(boundary_vals);
            contrast_score = max(sigmoid(mean_contrast * 6, 0.065, 6.0), ... % Increased from 0.06 to 0.065
                                 sigmoid(median_contrast * 6, 0.065, 6.0));
        else
            contrast_score = 0.31; % Increased from 0.32 to 0.31
        end
        
        % 9. VEIN ADJUSTMENT
        vein_penalty = sigmoid(vein_overlap_ratio, 0.43, 3.0) * 0.37; % Increased from 0.42/0.38 to 0.43/0.37
        vein_free_score = 1 - vein_penalty;
        
        % 10. COMBINED SCORE
        weights = [0.10, 0.15, 0.10, 0.35, 0.15, 0.05, 0.05, 0.05];
        
        % Ensure all scores are scalars
        intensity_score = double(intensity_score(1));
        dark_score = double(dark_score(1));
        local_dark_score = double(local_dark_score(1));
        size_score = double(size_score(1));
        very_dark_score = double(very_dark_score(1));
        vein_free_score = double(vein_free_score(1));
        contrast_score = double(contrast_score(1));
        homogeneity_score = double(homogeneity_score(1));
        
        region_score = weights(1)*intensity_score + ...
                       weights(2)*dark_score + ...
                       weights(3)*local_dark_score + ...
                       weights(4)*size_score + ...
                       weights(5)*very_dark_score + ...
                       weights(6)*vein_free_score + ...
                       weights(7)*contrast_score + ...
                       weights(8)*homogeneity_score;
        
        region_score = min(1, region_score + cluster_count_score);
        
        % Store scores
        region_scores(r, 1) = region_score;
        region_scores(r, 2) = intensity_score;
        region_scores(r, 3) = dark_score;
        region_scores(r, 4) = size_score;
        region_scores(r, 5) = very_dark_score;
        region_scores(r, 6) = vein_free_score;
        region_scores(r, 7) = contrast_score;
        region_scores(r, 8) = homogeneity_score;
        region_scores(r, 9) = vein_overlap_ratio;
        region_scores(r, 10) = max_area;
        
        % SLIGHTLY INCREASED THRESHOLDS for detection
        if (region_score > 0.43 && size_score > 0.33) || ... % Increased from 0.42/0.32 to 0.43/0.33
           (max_area > 82) || ... % Increased from 80 to 82
           (size_score > 0.49 && vein_free_score > 0.69) % Increased from 0.48/0.68 to 0.49/0.69
           
            hemorrhage_regions = [hemorrhage_regions, r];
            
            % Detection strategies - slightly more selective
            hem1 = img_double < dark_pixel_thresh1 & region_mask & strict_brain_mask;
            hem2 = img_double < very_low_percentile & region_mask & strict_brain_mask;
            large_dark = bwareaopen(all_dark, 29); % Increased from 28 to 29
            hem3 = large_dark;
            
            if any(neighborhood(:))
                hem4 = (img_double < neighborhood_mean * 0.85) & region_mask & strict_brain_mask; % Increased from 0.86 to 0.85
            else
                hem4 = false(size(img_double));
            end
            
            potential_hem = hem1 | hem2 | hem3 | hem4;
            potential_hem = potential_hem & ~vein_mask;
            
            potential_hem = bwareaopen(potential_hem, 9); % Kept at 9
            potential_hem = imclose(potential_hem, strel('disk', 1));
            
            % Slightly stricter filtering
            hem_props = regionprops(potential_hem, 'Area', 'Eccentricity', 'Solidity', 'Perimeter');
            if ~isempty(hem_props)
                filtered_hem = false(size(potential_hem));
                for k = 1:length(hem_props)
                    area = hem_props(k).Area;
                    ecc = hem_props(k).Eccentricity;
                    solidity = hem_props(k).Solidity;
                    perimeter = hem_props(k).Perimeter;
                    circularity = 4*pi*area / (perimeter^2 + eps);
                    
                    % SLIGHTLY STRICTER CRITERIA
                    keep_cluster = false;
                    
                    if area > 42 % Increased from 40 to 42
                        keep_cluster = true;
                    elseif area > 21 && area <= 42 % Increased lower bound from 20 to 21
                        if ecc < 0.71 && solidity > 0.74 % Stricter
                            keep_cluster = true;
                        elseif circularity > 0.47 % Increased from 0.48 to 0.47
                            keep_cluster = true;
                        end
                    elseif area > 11 && area <= 21 % Increased lower bound from 11 to 11 (kept same)
                        if very_dark_score > 0.37 && ecc < 0.61 && solidity > 0.79 % Stricter
                            keep_cluster = true;
                        end
                    end
                    
                    % Stricter vein-like check
                    is_vein_like = (ecc > 0.81 && area < 48) || (circularity < 0.15 && area < 58); % Stricter
                    
                    if keep_cluster && ~is_vein_like
                        CC_potential = bwconncomp(potential_hem);
                        if k <= CC_potential.NumObjects
                            filtered_hem = filtered_hem | ismember(labelmatrix(CC_potential), k);
                        end
                    end
                end
                potential_hem = filtered_hem;
            end
            
            hemorrhage_mask = hemorrhage_mask | potential_hem;
        end
    end
    
    % Final cleanup - Slightly more selective
    if any(hemorrhage_mask(:))
        hemorrhage_mask = bwareaopen(hemorrhage_mask, 17); % Increased from 16 to 17
        hemorrhage_mask = imclose(hemorrhage_mask, strel('disk', 1));
        hemorrhage_mask = imfill(hemorrhage_mask, 'holes');
        
        hemorrhage_mask = hemorrhage_mask & ~vein_mask;
        hemorrhage_mask = hemorrhage_mask & strict_brain_mask;
        
        boundary_dist = bwdist(~strict_brain_mask);
        large_enough = hemorrhage_mask > 58; % Increased from 55 to 58
        hemorrhage_mask = hemorrhage_mask & (boundary_dist > 3 | large_enough);
        
        % FINAL QUALITY CHECK - Slightly stricter
        final_props = regionprops(hemorrhage_mask, 'Area', 'Eccentricity', 'Solidity', 'Perimeter');
        if ~isempty(final_props)
            final_filtered = false(size(hemorrhage_mask));
            for k = 1:length(final_props)
                area = final_props(k).Area;
                ecc = final_props(k).Eccentricity;
                solidity = final_props(k).Solidity;
                perimeter = final_props(k).Perimeter;
                circularity = 4*pi*area / (perimeter^2 + eps);
                
                if area > 58 % Increased from 55 to 58
                    if solidity > 0.61 % Increased from 0.62 to 0.61
                        final_filtered = final_filtered | ismember(labelmatrix(bwconncomp(hemorrhage_mask)), k);
                    end
                elseif area > 30 && area <= 58 % Increased lower bound from 28 to 30
                    if ecc < 0.71 && solidity > 0.71 && circularity > 0.32 % Slightly stricter
                        final_filtered = final_filtered | ismember(labelmatrix(bwconncomp(hemorrhage_mask)), k);
                    end
                elseif area > 17 && area <= 30 % Increased lower bound from 16 to 17
                    if ecc < 0.51 && solidity > 0.79 && circularity > 0.41 % Slightly stricter
                        final_filtered = final_filtered | ismember(labelmatrix(bwconncomp(hemorrhage_mask)), k);
                    end
                end
            end
            hemorrhage_mask = final_filtered;
        end
    end
    
    fprintf('  Detected %d region(s) with potential dark hemorrhage (slightly increased threshold)\n', length(hemorrhage_regions));
end

% HELPER FUNCTION: Sigmoid for scoring
function y = sigmoid(x, threshold, slope)
    y = 1 ./ (1 + exp(-slope * (x - threshold)));
end

% PROCESSING PIPELINE - WITH DARK HEMORRHAGE DETECTION
fprintf('\n%s\n', repmat('-', 1, 70));
fprintf('SWI VESSEL EXTRACTION & HEMORRHAGE DETECTION (WITH VEIN DIFFERENTIATION)\n');
fprintf('%s\n', repmat('-', 1, 70));

% Step 1: Skull stripping
fprintf('\nStep 1: Removing skull...\n');
brain_mask = skull_strip_swi(original);
brain_only = original;
brain_only(~brain_mask) = 0;

% Step 2: Create brain interior mask (remove edge) - STRICTER
fprintf('Step 2: Creating brain interior mask (stricter edge removal)...\n');
interior_mask = create_brain_interior_mask(brain_mask, 10);

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

% Step 5: Get region masks for hemorrhage detection
[region_coords, region_masks, ~] = split_into_focused_regions(interior_mask, 12);

% Step 6: VESSEL EXTRACTION (INTERIOR ONLY)
fprintf('Step 5: Performing vessel extraction (interior only)...\n');
[vessel_mask, vesselness_map] = extract_brain_vessels(brain_enhanced, brain_mask, interior_mask);

% Step 7: VEIN DETECTION (to differentiate from hemorrhage)
fprintf('Step 6: Detecting veins for differentiation...\n');
vein_mask = detect_veins(original, brain_mask, interior_mask);

% Step 8: DARK HEMORRHAGE DETECTION (BEST PERFORMING)
fprintf('Step 7: Detecting dark hemorrhage signs (increased sensitivity)...\n');
[hemorrhage_mask, region_scores, hemorrhage_regions] = detect_dark_hemorrhage(original, brain_mask, interior_mask, region_masks, vein_mask);

% CREATE VISUALIZATIONS WITH DARK HEMORRHAGE INDICATION

% 1. ORIGINAL IMAGE
original_display = original;

% 2. SPLIT BRAIN REGIONS (with hemorrhage-highlighted regions)
regions_display = original;
regions_display = im2double(regions_display);
regions_display = repmat(regions_display, [1,1,3]); % Convert to RGB

% Draw region boundaries in different colors - 12 colors
colors = lines(12);
for r = 1:length(region_coords)
    coords = region_coords{r};
    % Determine region color (red if hemorrhage detected, otherwise normal)
    if ismember(r, hemorrhage_regions)
        region_color = [1, 0, 0]; % Red for hemorrhage regions
    else
        region_color = colors(r, :);
    end
    
    % Draw rectangle
    for x = coords(3):coords(4)
        if x >= 1 && x <= size(regions_display,2)
            if coords(1) >= 1 && coords(1) <= size(regions_display,1)
                regions_display(coords(1), x, :) = reshape(region_color, [1,1,3]);
            end
            if coords(2) >= 1 && coords(2) <= size(regions_display,1)
                regions_display(coords(2), x, :) = reshape(region_color, [1,1,3]);
            end
        end
    end
    for y = coords(1):coords(2)
        if y >= 1 && y <= size(regions_display,1)
            if coords(3) >= 1 && coords(3) <= size(regions_display,2)
                regions_display(y, coords(3), :) = reshape(region_color, [1,1,3]);
            end
            if coords(4) >= 1 && coords(4) <= size(regions_display,2)
                regions_display(y, coords(4), :) = reshape(region_color, [1,1,3]);
            end
        end
    end
end

% 3. EXTRACTED BRAIN
brain_display = brain_only;
brain_display = im2double(brain_display);
brain_display = repmat(brain_display, [1,1,3]);
% Set background to white
for c = 1:3
    channel = brain_display(:,:,c);
    channel(~brain_mask) = 1;
    brain_display(:,:,c) = channel;
end

% 4. VEIN OVERLAY (for visualization)
vein_display = im2double(original);
vein_display = repmat(vein_display, [1,1,3]);
% Highlight veins in BLUE
if any(vein_mask(:))
    for c = 1:3
        channel = vein_display(:,:,c);
        if c == 3 % Blue channel
            channel(vein_mask) = 1;
        else
            channel(vein_mask) = 0;
        end
        vein_display(:,:,c) = channel;
    end
end

% 5. DARK HEMORRHAGE OVERLAY (highlight in YELLOW)
hemorrhage_display = im2double(original);
hemorrhage_display = repmat(hemorrhage_display, [1,1,3]);
% Highlight dark hemorrhage in YELLOW
if any(hemorrhage_mask(:))
    for c = 1:3
        channel = hemorrhage_display(:,:,c);
        if c == 1 || c == 2 % Yellow = Red + Green
            channel(hemorrhage_mask) = 1;
        else
            channel(hemorrhage_mask) = 0;
        end
        hemorrhage_display(:,:,c) = channel;
    end
end

% 6. RESULT - SEGMENTED BLOOD VESSELS + DARK HEMORRHAGE (with vein differentiation)
result_display = 255 * ones(size(brain_enhanced), 'uint8');
result_display = repmat(result_display, [1,1,3]); % RGB
result_display = im2double(result_display);
% Set vessels to BLACK
for c = 1:3
    channel = result_display(:,:,c);
    channel(vessel_mask) = 0;
    result_display(:,:,c) = channel;
end
% Highlight veins in BLUE
if any(vein_mask(:))
    for c = 1:3
        channel = result_display(:,:,c);
        if c == 3 % Blue channel
            channel(vein_mask) = 1;
        else
            channel(vein_mask) = 0.3; % Dim other channels
        end
        result_display(:,:,c) = channel;
    end
end
% Highlight dark hemorrhage in YELLOW (excludes veins)
if any(hemorrhage_mask(:))
    for c = 1:3
        channel = result_display(:,:,c);
        if c == 1 || c == 2 % Yellow
            channel(hemorrhage_mask) = 1;
        else
            channel(hemorrhage_mask) = 0;
        end
        result_display(:,:,c) = channel;
    end
end
% Set background to white
for c = 1:3
    channel = result_display(:,:,c);
    channel(~brain_mask) = 1;
    result_display(:,:,c) = channel;
end

% MAIN VISUALIZATION - 6-PANEL PLOT
figure('Name', 'SWI Vessel Extraction & Dark Hemorrhage Detection (Best Performance)', ...
       'Position', [50, 50, 1800, 1000], ...
       'Color', 'white');

% Panel 1: Original Image
subplot(2,3,1);
imshow(original_display);
title('A. ORIGINAL SWI (Dark = Blood)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

% Panel 2: Split Brain Regions
subplot(2,3,2);
imshow(regions_display);
title('B. BRAIN REGIONS (12 Regions)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;
if ~isempty(hemorrhage_regions)
    text(0.02, 0.98, sprintf('Hemorrhage regions: %s', mat2str(hemorrhage_regions)), ...
         'Units', 'normalized', 'VerticalAlignment', 'top', 'Color', 'red', ...
         'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'yellow');
end

% Panel 3: Vein Detection
subplot(2,3,3);
imshow(vein_display);
title('C. VEIN DETECTION (Blue)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;
text(0.02, 0.98, sprintf('Veins: %d pixels', sum(vein_mask(:))), ...
     'Units', 'normalized', 'VerticalAlignment', 'top', 'Color', 'white', ...
     'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'blue');

% Panel 4: Dark Hemorrhage Detection
subplot(2,3,4);
imshow(hemorrhage_display);
title('D. HEMORRHAGE DETECTION (Yellow)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;
if any(hemorrhage_mask(:))
    text(0.02, 0.98, sprintf('Hemorrhage: %d pixels', sum(hemorrhage_mask(:))), ...
         'Units', 'normalized', 'VerticalAlignment', 'top', 'Color', 'red', ...
         'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'yellow');
else
    text(0.02, 0.98, 'NO HEMORRHAGE DETECTED', 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'Color', 'green', 'FontSize', 12, ...
         'FontWeight', 'bold', 'BackgroundColor', 'white');
end

% Panel 5: Result - Differentiated
subplot(2,3,5);
imshow(result_display);
title('E. DIFFERENTIATED: Veins (Blue) vs Hemorrhage (Yellow)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;
text(0.02, 0.98, 'Black=Vessels, Blue=Veins, Yellow=Hemorrhage', 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'Color', 'black', 'FontSize', 9, ...
     'FontWeight', 'bold', 'BackgroundColor', 'white');

% Panel 6: Statistics Panel
subplot(2,3,6);
axis off;
text(0.1, 0.95, 'DIFFERENTIATION RESULTS', 'FontSize', 14, 'FontWeight', 'bold');

if any(hemorrhage_mask(:))
    text(0.1, 0.85, 'Hemorrhage: YES', 'FontSize', 12, 'Color', [1 0 0]);
else
    text(0.1, 0.85, 'Hemorrhage: NO', 'FontSize', 12, 'Color', [0 0.5 0]);
end

text(0.1, 0.75, sprintf('Veins: %d pixels', sum(vein_mask(:))), 'FontSize', 11, 'Color', [0 0 1]);
text(0.1, 0.65, sprintf('Hemorrhage: %d pixels', sum(hemorrhage_mask(:))), 'FontSize', 11, 'Color', [1 0 0]);

if ~isempty(hemorrhage_regions)
    text(0.1, 0.55, 'Affected regions:', 'FontSize', 11, 'FontWeight', 'bold');
    text(0.15, 0.48, mat2str(hemorrhage_regions), 'FontSize', 10);
end

text(0.1, 0.35, 'VESSEL STATISTICS', 'FontSize', 12, 'FontWeight', 'bold');
vessel_density = 100 * sum(vessel_mask(:)) / sum(interior_mask(:));
text(0.1, 0.25, sprintf('Vessel density: %.2f%%', vessel_density), 'FontSize', 11);
text(0.1, 0.15, sprintf('Total vessels: %d', sum(vessel_mask(:))), 'FontSize', 11);

text(0.1, 0.05, 'Veins excluded from hemorrhage', 'FontSize', 10, 'Color', [0 0.5 0]);

sgtitle('SWI Analysis: Vessel Extraction & Hemorrhage Detection (Best Performance)', ...
        'FontSize', 16, 'FontWeight', 'bold');

% SUPPLEMENTARY FIGURE 1 - Size Analysis
figure('Name', 'Hemorrhage Size Analysis', ...
       'Position', [200, 200, 1400, 500], ...
       'Color', 'white');

% Show size scores with threshold
subplot(1,3,1);
if size(region_scores, 1) >= 12
    % Create bar data with size score and normalized max area
    bar_data = [region_scores(1:12, 4), region_scores(1:12, 10)/max(region_scores(:,10)+eps)*100];
    b = bar(bar_data);
    b(1).FaceColor = [0.2 0.6 1]; % Blue for size score
    b(2).FaceColor = [1 0.6 0.2]; % Orange for normalized area
    hold on;
    plot([0,13], [0.33, 0.33], 'r--', 'LineWidth', 2, 'Color', [0.8 0 0]); % UPDATED: 0.32 -> 0.33
    legend({'Size Score', 'Area % of Max', 'Detection Threshold'}, 'Location', 'northwest', 'FontSize', 9);
    xlabel('Region Number', 'FontSize', 11);
    ylabel('Score / Percentage', 'FontSize', 11);
    title('Size Analysis for Hemorrhage Detection', 'FontWeight', 'bold', 'FontSize', 12);
    xlim([0, 13]);
    ylim([0, 100]);
    grid on;
    set(gca, 'GridAlpha', 0.3);
    
    % Add region numbers on top of significant bars - UPDATED threshold from 0.28 to 0.29
    for i = 1:12
        if region_scores(i, 4) > 0.29
            text(i, region_scores(i, 4) + 3, num2str(i), ...
                 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold', 'Color', 'b');
        end
    end
end

% Show vein vs hemorrhage overlay with improved visualization
subplot(1,3,2);
% Create RGB overlay with better contrast
overlay = im2double(original);
overlay = repmat(overlay, [1,1,3]);
overlay = overlay * 0.7; % Darken background for better contrast

% Veins in bright blue
if any(vein_mask(:))
    for c = 1:3
        channel = overlay(:,:,c);
        if c == 3 % Blue channel
            channel(vein_mask) = 1;
        else
            channel(vein_mask) = 0.2;
        end
        overlay(:,:,c) = channel;
    end
end

% Hemorrhage in bright yellow
if any(hemorrhage_mask(:))
    for c = 1:3
        channel = overlay(:,:,c);
        if c == 1 || c == 2 % Yellow = Red + Green
            channel(hemorrhage_mask) = 1;
        else
            channel(hemorrhage_mask) = 0;
        end
        overlay(:,:,c) = channel;
    end
end

imshow(overlay);
title('Veins (Blue) vs Hemorrhage (Yellow)', 'FontWeight', 'bold', 'FontSize', 12);
axis image;

% Add legend on the image
rectangle('Position', [10, 10, 30, 20], 'FaceColor', [0 0 1], 'EdgeColor', 'none');
text(45, 20, 'Veins', 'Color', 'white', 'FontWeight', 'bold', 'FontSize', 10);
rectangle('Position', [10, 40, 30, 20], 'FaceColor', [1 1 0], 'EdgeColor', 'none');
text(45, 50, 'Hemorrhage', 'Color', 'black', 'FontWeight', 'bold', 'FontSize', 10);

% Show comprehensive statistics
subplot(1,3,3);
axis off;

% Create a nice box for statistics
rectangle('Position', [0.1, 0.1, 0.8, 0.8], 'Curvature', 0.1, ...
          'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1.5, 'FaceColor', [0.95 0.95 0.95]);

text(0.2, 0.85, 'DETECTION SUMMARY', 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.2 0.2 0.5]);

% Hemorrhage statistics
text(0.2, 0.72, sprintf('Hemorrhage pixels: %d', sum(hemorrhage_mask(:))), ...
     'FontSize', 13, 'Color', [0.8 0.4 0], 'FontWeight', 'bold');

if any(hemorrhage_mask(:))
    text(0.2, 0.62, sprintf('Largest cluster: %.0f pixels', max(region_scores(:,10))), ...
         'FontSize', 13, 'Color', [0.8 0.4 0]);
    text(0.2, 0.52, sprintf('Affected regions: %d', length(hemorrhage_regions)), ...
         'FontSize', 13, 'Color', [0.8 0.4 0]);
    if ~isempty(hemorrhage_regions)
        text(0.2, 0.42, sprintf('Region numbers: %s', mat2str(hemorrhage_regions)), ...
             'FontSize', 11, 'Color', [0.4 0.4 0.4]);
    end
else
    text(0.2, 0.62, 'No hemorrhage detected', 'FontSize', 13, 'Color', [0 0.5 0], 'FontWeight', 'bold');
end

% Vein statistics
text(0.2, 0.27, sprintf('Vein pixels: %d', sum(vein_mask(:))), ...
     'FontSize', 12, 'Color', [0 0 0.8]);

% Quality indicator
if any(hemorrhage_mask(:))
    quality_text = 'Quality filtered';
    quality_color = [0 0.8 0];
else
    quality_text = 'No detection to filter';
    quality_color = [0.5 0.5 0.5];
end
text(0.2, 0.15, quality_text, 'FontSize', 12, 'Color', quality_color, 'FontWeight', 'bold');

% SUPPLEMENTARY FIGURE 2 - Quality Analysis
figure('Name', 'Hemorrhage Quality Analysis', ...
       'Position', [200, 200, 1400, 500], ...
       'Color', 'white');

% Show size scores with quality threshold
subplot(1,3,1);
if size(region_scores, 1) >= 12
    % Plot size scores with color coding for detected regions
    size_scores = region_scores(1:12, 4);
    b = bar(size_scores);
    
    % Color bars based on detection status
    hold on;
    for i = 1:12
        if ismember(i, hemorrhage_regions)
            b.FaceColor = 'flat';
            b.CData(i,:) = [1 0.5 0]; % Orange for detected regions
        else
            b.FaceColor = 'flat';
            b.CData(i,:) = [0.5 0.5 1]; % Light blue for non-detected
        end
    end
    
    plot([0,13], [0.33, 0.33], 'g--', 'LineWidth', 2, 'Color', [0 0.8 0]); % UPDATED: 0.32 -> 0.33
    xlabel('Region Number', 'FontSize', 11);
    ylabel('Size Score', 'FontSize', 11);
    title('Size Score by Region (Quality Threshold)', 'FontWeight', 'bold', 'FontSize', 12);
    xlim([0, 13]);
    ylim([0, 1]);
    grid on;
    set(gca, 'GridAlpha', 0.3);
    
    % Add threshold value - UPDATED text
    text(11, 0.36, 'Threshold: 0.33', 'FontSize', 9, 'Color', [0 0.8 0]);
    
    % Legend
    text(1, 0.9, '■ Detected', 'Color', [1 0.5 0], 'FontSize', 10, 'FontWeight', 'bold');
    text(4, 0.9, '■ Not Detected', 'Color', [0.5 0.5 1], 'FontSize', 10, 'FontWeight', 'bold');
end

% Show final hemorrhage overlay with brain mask outline
subplot(1,3,2);
overlay = im2double(original);
overlay = repmat(overlay, [1,1,3]);
overlay = overlay * 0.8; % Slightly darken

% Add brain outline
brain_perim = bwperim(brain_mask);
for c = 1:3
    channel = overlay(:,:,c);
    channel(brain_perim) = 0.5;
    overlay(:,:,c) = channel;
end

% Hemorrhage in yellow
if any(hemorrhage_mask(:))
    for c = 1:3
        channel = overlay(:,:,c);
        if c == 1 || c == 2
            channel(hemorrhage_mask) = 1;
        end
        overlay(:,:,c) = channel;
    end
end

imshow(overlay);
title('Final Hemorrhage Detection (Brain Outline)', 'FontWeight', 'bold', 'FontSize', 12);
axis image;

% Show detailed statistics with quality metrics
subplot(1,3,3);
axis off;

% Create a nice box for statistics
rectangle('Position', [0.1, 0.1, 0.8, 0.8], 'Curvature', 0.1, ...
          'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1.5, 'FaceColor', [0.98 0.98 0.98]);

text(0.2, 0.9, 'QUALITY METRICS', 'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.2 0.5 0.2]);

if any(hemorrhage_mask(:))
    % Detection metrics
    text(0.2, 0.78, sprintf('Total hemorrhage area: %d px', sum(hemorrhage_mask(:))), 'FontSize', 12);
    text(0.2, 0.68, sprintf('Number of clusters: %d', length(regionprops(hemorrhage_mask))), 'FontSize', 12);
    text(0.2, 0.58, sprintf('Largest cluster: %.0f px', max(region_scores(:,10))), 'FontSize', 12);
    
    % Quality scores for top regions
    text(0.2, 0.45, 'Top Region Quality:', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.4 0.4 0.4]);
    
    [sorted_scores, sorted_idx] = sort(region_scores(:,1), 'descend');
    for i = 1:min(3, sum(sorted_scores > 0))
        r = sorted_idx(i);
        y_pos = 0.38 - i*0.07;
        text(0.25, y_pos, sprintf('Region %d: Score=%.2f | Size=%.2f', ...
             r, region_scores(r,1), region_scores(r,4)), 'FontSize', 10);
    end
    
    % Overall quality assessment - UPDATED thresholds for slightly increased sensitivity
    avg_size_score = mean(region_scores(hemorrhage_regions, 4));
    if avg_size_score > 0.53
        quality = 'EXCELLENT';
        quality_color = [0 0.8 0];
    elseif avg_size_score > 0.39
        quality = 'GOOD';
        quality_color = [0.8 0.8 0];
    elseif avg_size_score > 0.29
        quality = 'FAIR';
        quality_color = [1 0.6 0];
    else
        quality = 'MARGINAL';
        quality_color = [1 0.3 0.3];
    end
    
    text(0.2, 0.12, sprintf('Quality: %s', quality), 'FontSize', 14, ...
         'Color', quality_color, 'FontWeight', 'bold');
else
    text(0.2, 0.7, 'No hemorrhage detected', 'FontSize', 14, 'Color', [0 0.5 0], 'FontWeight', 'bold');
    text(0.2, 0.5, 'All regions normal', 'FontSize', 12, 'Color', [0.4 0.4 0.4]);
    text(0.2, 0.3, 'Quality check passed', 'FontSize', 12, 'Color', [0 0.5 0]);
end

% Add a small note about updated thresholds at the bottom of the figure
annotation('textbox', [0.02, 0.02, 0.3, 0.03], ...
           'String', 'Note: Detection threshold slightly increased to 0.33', ...
           'FontSize', 9, 'Color', [0.5 0.5 0.5], 'EdgeColor', 'none', 'HorizontalAlignment', 'left');

fprintf('%s\n', repmat('-', 1, 70));
fprintf('\nProcessing Complete!\n');
fprintf('Generated:\n');
fprintf('  1. Figure 1: 6-Panel Pipeline with Vein Differentiation\n');
fprintf('  2. Figure 2: Hemorrhage Size Analysis (Threshold: 0.33)\n');
fprintf('  3. Figure 3: Hemorrhage Quality Analysis (Threshold: 0.33)\n');