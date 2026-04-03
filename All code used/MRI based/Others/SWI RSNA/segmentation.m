% =========================================================================
% SWI BRAIN ANEURYSM DETECTION - WITH REGION SPLITTING (12 REGIONS)
% =========================================================================
% Based on your vessel extraction code with region splitting and blending
% =========================================================================

clear all; clc; close all;

% =========================================================================
% LOAD IMAGE
% =========================================================================
files = [dir('*.jpg'); dir('*.png'); dir('*.tif'); dir('*.bmp'); dir('*.dcm')];
if isempty(files)
    error('No image files found. Place an SWI image in current folder.');
end

img = imread(files(randi(length(files))).name);

if size(img, 3) == 3
    original = rgb2gray(img);
else
    original = img;
end

% =========================================================================
% FUNCTION: Skull Stripping
% =========================================================================
function brain_mask = skull_strip_swi(mri_image)
    img_double = im2double(mri_image);
    
    level = graythresh(img_double);
    
    mask1 = imbinarize(img_double, level * 0.6);
    mask2 = imbinarize(img_double, level * 0.7);
    mask3 = imbinarize(img_double, level * 0.8);
    
    initial_mask = mask1 | mask2 | mask3;
    initial_mask = bwareaopen(initial_mask, 500);
    initial_mask = imfill(initial_mask, 'holes');
    
    CC = bwconncomp(initial_mask);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    [~, idx] = max(numPixels);
    brain_mask = false(size(initial_mask));
    brain_mask(CC.PixelIdxList{idx}) = true;
    
    se = strel('disk', 3);
    brain_mask = imerode(brain_mask, se);
    brain_mask = imfill(brain_mask, 'holes');
    brain_mask = imclose(brain_mask, strel('disk', 5));
    brain_mask = bwmorph(brain_mask, 'spur', 5);
    brain_mask = imdilate(brain_mask, strel('disk', 2));
end

% =========================================================================
% FUNCTION: Create Brain Interior Mask
% =========================================================================
function interior_mask = create_brain_interior_mask(brain_mask, erosion_pixels)
    if nargin < 2
        erosion_pixels = 8;
    end
    
    se = strel('disk', erosion_pixels);
    interior_mask = imerode(brain_mask, se);
    
    boundary_mask = brain_mask & ~interior_mask;
    
    fprintf('  Brain interior: %d pixels (%.1f%% of brain)\n', ...
        sum(interior_mask(:)), 100*sum(interior_mask(:))/sum(brain_mask(:)));
    fprintf('  Brain boundary (suppressed): %d pixels\n', sum(boundary_mask(:)));
end

% =========================================================================
% FUNCTION: Wavelet Denoising
% =========================================================================
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

% =========================================================================
% FUNCTION: Split Brain into Smaller Non-Overlapping Regions (12 Regions)
% =========================================================================
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

% =========================================================================
% FUNCTION: Create Blending Weights
% =========================================================================
function blending_weights = create_blending_weights(region_masks, sigma)
    if nargin < 2
        sigma = 20;
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

% =========================================================================
% FUNCTION: Aneurysm-Specific Vessel Analysis (Modified for regions)
% =========================================================================
function vessel_properties = extract_vessel_properties(img, interior_mask)
    fprintf('  Extracting vessel properties for aneurysm detection...\n');
    
    img_double = im2double(img);
    img_inverted = 1 - img_double;
    
    % Multi-scale vessel enhancement
    scales = [1, 1.5, 2, 2.5, 3, 3.5, 4];
    vessel_response = zeros(size(img_double));
    vessel_scale = zeros(size(img_double));
    vessel_eccentricity = zeros(size(img_double));
    
    for s = 1:length(scales)
        sigma = scales(s);
        
        % Create Gaussian derivatives
        n = ceil(4*sigma);
        x = -n:n;
        y = -n:n;
        [X, Y] = meshgrid(x, y);
        
        G = exp(-(X.^2 + Y.^2) / (2*sigma^2));
        G = G / sum(G(:));
        
        Gxx = (X.^2 / sigma^4 - 1/sigma^2) .* G;
        Gxy = (X.*Y / sigma^4) .* G;
        Gyy = (Y.^2 / sigma^4 - 1/sigma^2) .* G;
        
        % Compute Hessian
        Dxx = conv2(img_inverted, Gxx, 'same');
        Dxy = conv2(img_inverted, Gxy, 'same');
        Dyy = conv2(img_inverted, Gyy, 'same');
        
        % Compute eigenvalues
        trace_H = Dxx + Dyy;
        det_H = Dxx .* Dyy - Dxy.^2;
        discriminant = sqrt(max(0, (trace_H/2).^2 - det_H));
        
        lambda1 = trace_H/2 + discriminant;
        lambda2 = trace_H/2 - discriminant;
        
        % Vesselness measure (Frangi)
        Rb = abs(lambda1) ./ (abs(lambda2) + eps);
        S = sqrt(lambda1.^2 + lambda2.^2);
        
        Beta = 0.5;
        Gamma = 0.25;
        
        response = exp(-Rb.^2 / (2*Beta^2)) .* (1 - exp(-S.^2 / (2*Gamma^2)));
        response(lambda2 >= 0) = 0;
        
        % Update maximum response
        update_mask = response > vessel_response;
        vessel_response(update_mask) = response(update_mask);
        vessel_scale(update_mask) = sigma;
        vessel_eccentricity(update_mask) = abs(lambda1(update_mask) ./ (lambda2(update_mask) + eps));
    end
    
    % Normalize responses
    vessel_response = mat2gray(vessel_response);
    vessel_response(~interior_mask) = 0;
    
    % Adaptive thresholding for vessel mask
    interior_vals = vessel_response(interior_mask);
    if ~isempty(interior_vals)
        thresh = graythresh(interior_vals) * 0.7;
        vessel_mask = vessel_response > thresh;
        
        % Clean up vessel mask
        vessel_mask = bwareaopen(vessel_mask, 10);
        vessel_mask = imclose(vessel_mask, strel('disk', 1));
        vessel_mask = vessel_mask & interior_mask;
    else
        vessel_mask = false(size(vessel_response));
    end
    
    % Store properties
    vessel_properties = struct();
    vessel_properties.response = vessel_response;
    vessel_properties.scale = vessel_scale;
    vessel_properties.eccentricity = vessel_eccentricity;
    vessel_properties.mask = vessel_mask;
    
    fprintf('  Extracted %d vessel pixels\n', sum(vessel_mask(:)));
end

% =========================================================================
% FUNCTION: Texture Feature Extraction (for aneurysm detection)
% =========================================================================
function texture_map = extract_vessel_texture(img, region_mask)
    img_double = im2double(img);
    texture_map = zeros(size(img_double));
    
    window_size = 7;
    padded_img = padarray(img_double, [floor(window_size/2), floor(window_size/2)], 'symmetric');
    
    [rows, cols] = size(img_double);
    margin = floor(window_size/2);
    
    for i = margin+1:rows-margin
        for j = margin+1:cols-margin
            if region_mask(i, j)
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

% =========================================================================
% FUNCTION: Analyze Each Region for Aneurysm Detection
% =========================================================================
function [blended_aneurysm_map, region_candidates, region_scores_all] = analyze_regions_for_aneurysms(img, interior_mask, num_regions)
    % Split brain interior into regions
    [region_coords, region_masks, ~] = split_into_focused_regions(interior_mask, num_regions);
    num_valid_regions = length(region_masks);
    
    % Create blending weights
    fprintf('    Creating blending weights...\n');
    blending_weights = create_blending_weights(region_masks, 20);
    
    % Initialize results
    region_aneurysm_maps = zeros([size(img), num_valid_regions]);
    region_candidates = cell(num_valid_regions, 1);
    region_scores_all = cell(num_valid_regions, 1);
    
    % Analyze each region
    for r = 1:num_valid_regions
        fprintf('    Analyzing region %d/%d for aneurysms...\n', r, num_valid_regions);
        
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
        
        % Extract vessel properties in this region
        region_vessel_props = extract_vessel_properties(region_enhanced, region_mask);
        
        % Extract texture features
        region_texture = extract_vessel_texture(region_enhanced, region_mask);
        
        % Find aneurysm candidates in this region
        [region_candidates_mask, region_scores] = find_aneurysm_candidates_in_region(...
            region_vessel_props, region_texture, region_mask);
        
        % Store results
        region_aneurysm_maps(:,:,r) = region_candidates_mask .* region_mask;
        region_candidates{r} = region_candidates_mask;
        region_scores_all{r} = region_scores;
    end
    
    % Blend results
    fprintf('    Blending region results...\n');
    blended_aneurysm_map = zeros(size(img));
    
    for r = 1:num_valid_regions
        blended_aneurysm_map = blended_aneurysm_map + region_aneurysm_maps(:,:,r) .* blending_weights(:,:,r);
    end
    
    % Apply threshold
    blended_aneurysm_map = blended_aneurysm_map > 0.3;
    blended_aneurysm_map = blended_aneurysm_map & interior_mask;
    
    % Clean up
    blended_aneurysm_map = bwareaopen(blended_aneurysm_map, 15);
    blended_aneurysm_map = imclose(blended_aneurysm_map, strel('disk', 1));
    
    fprintf('  Found %d aneurysm candidates across all regions\n', sum(blended_aneurysm_map(:)));
end

% =========================================================================
% FUNCTION: Find Aneurysm Candidates in a Single Region
% =========================================================================
function [candidate_mask, scores] = find_aneurysm_candidates_in_region(vessel_props, texture_map, region_mask)
    
    vessel_mask = vessel_props.mask;
    vessel_response = vessel_props.response;
    
    candidate_mask = false(size(vessel_mask));
    scores = [];
    
    % Label connected components
    CC = bwconncomp(vessel_mask);
    
    if CC.NumObjects == 0
        return;
    end
    
    % Get component properties
    props = regionprops(CC, 'Area', 'Perimeter', 'Eccentricity', ...
                        'Solidity', 'MajorAxisLength', 'MinorAxisLength', ...
                        'PixelIdxList');
    
    for i = 1:CC.NumObjects
        component = false(size(vessel_mask));
        component(CC.PixelIdxList{i}) = true;
        
        area = props(i).Area;
        perimeter = props(i).Perimeter;
        eccentricity = props(i).Eccentricity;
        solidity = props(i).Solidity;
        major_axis = props(i).MajorAxisLength;
        minor_axis = props(i).MinorAxisLength;
        
        % Skip very small components
        if area < 20
            continue;
        end
        
        circularity = 4*pi*area / (perimeter^2 + eps);
        
        % Get texture values in this component
        texture_vals = texture_map(component);
        mean_texture = mean(texture_vals);
        
        % Get vessel response values
        response_vals = vessel_response(component);
        mean_response = mean(response_vals);
        std_response = std(response_vals);
        
        % Look for focal dilations using distance transform
        dist_transform = bwdist(~component);
        dist_transform(~component) = 0;
        vessel_thickness = dist_transform * 2;
        thick_pixels = vessel_thickness > 0 & component;
        
        dilation_score = 0;
        if any(thick_pixels(:))
            mean_thickness = mean(vessel_thickness(thick_pixels));
            std_thickness = std(vessel_thickness(thick_pixels));
            
            if mean_thickness > 0
                dilation_score = min(1, std_thickness / mean_thickness);
            end
            
            % Find thick regions
            thickness_threshold = mean_thickness + 1.5 * std_thickness;
            thick_regions = vessel_thickness > thickness_threshold;
            thick_CC = bwconncomp(thick_regions & component);
            
            for j = 1:thick_CC.NumObjects
                thick_area = length(thick_CC.PixelIdxList{j});
                if thick_area > 5 && thick_area < area * 0.4
                    candidate_mask(thick_CC.PixelIdxList{j}) = true;
                end
            end
        end
        
        % Calculate aneurysm scores
        shape_score = (circularity * 0.6 + (1 - eccentricity) * 0.4);
        intensity_score = mean_response;
        texture_score = mean_texture;
        
        % Combined score
        weights = [0.35, 0.25, 0.20, 0.20];  % Shape, dilation, intensity, texture
        total_score = weights(1)*shape_score + ...
                      weights(2)*dilation_score + ...
                      weights(3)*intensity_score + ...
                      weights(4)*texture_score;
        
        % Store scores if this component has candidates
        if any(candidate_mask(component))
            scores = [scores; total_score, shape_score, dilation_score, ...
                     intensity_score, texture_score, area, circularity];
        end
    end
end

% =========================================================================
% FUNCTION: Classify Aneurysms
% =========================================================================
function [aneurysm_mask, aneurysm_scores, aneurysm_regions] = classify_aneurysms(blended_candidates, interior_mask)
    fprintf('  Classifying aneurysm candidates...\n');
    
    aneurysm_mask = false(size(interior_mask));
    aneurysm_scores = [];
    aneurysm_regions = [];
    
    if ~any(blended_candidates(:))
        fprintf('  No candidates to classify\n');
        return;
    end
    
    % Label candidate components
    CC = bwconncomp(blended_candidates);
    
    if CC.NumObjects == 0
        return;
    end
    
    % Get candidate properties
    candidate_props = regionprops(CC, 'Area', 'Perimeter', 'Eccentricity', ...
                                   'Solidity', 'MajorAxisLength', 'MinorAxisLength');
    
    for i = 1:CC.NumObjects
        area = candidate_props(i).Area;
        perimeter = candidate_props(i).Perimeter;
        eccentricity = candidate_props(i).Eccentricity;
        solidity = candidate_props(i).Solidity;
        
        circularity = 4*pi*area / (perimeter^2 + eps);
        
        % Classification criteria
        size_ok = area > 20 && area < 400;
        shape_ok = circularity > 0.35 && eccentricity < 0.8;
        boundary_ok = solidity > 0.7;
        
        % Not too elongated
        not_elongated = eccentricity < 0.85;
        
        criteria_count = sum([size_ok, shape_ok, boundary_ok, not_elongated]);
        
        if area > 100
            criteria_threshold = 3;
        elseif area > 40
            criteria_threshold = 3;
        else
            criteria_threshold = 2;
        end
        
        if criteria_count >= criteria_threshold
            aneurysm_mask(CC.PixelIdxList{i}) = true;
            aneurysm_regions = [aneurysm_regions, i];
            aneurysm_scores = [aneurysm_scores; area, circularity, eccentricity, solidity];
        end
    end
    
    % Post-processing
    if any(aneurysm_mask(:))
        aneurysm_mask = bwareaopen(aneurysm_mask, 15);
        aneurysm_mask = imclose(aneurysm_mask, strel('disk', 1));
        aneurysm_mask = aneurysm_mask & interior_mask;
        
        % Remove boundary-touching candidates
        boundary_dist = bwdist(~interior_mask);
        aneurysm_mask = aneurysm_mask & (boundary_dist > 5);
    end
    
    fprintf('  Classified %d regions as aneurysms\n', length(aneurysm_regions));
end

% =========================================================================
% FUNCTION: Aneurysm Risk Assessment
% =========================================================================
function risk_assessment = assess_aneurysm_risk(aneurysm_mask, aneurysm_scores)
    risk_assessment = struct();
    risk_assessment.has_aneurysm = false;
    risk_assessment.risk_level = 'None';
    risk_assessment.risk_score = 0;
    risk_assessment.recommendation = 'No aneurysms detected';
    risk_assessment.num_aneurysms = 0;
    risk_assessment.max_size_pixels = 0;
    risk_assessment.total_area_pixels = 0;
    
    if ~any(aneurysm_mask(:)) || isempty(aneurysm_scores)
        return;
    end
    
    CC = bwconncomp(aneurysm_mask);
    num_aneurysms = CC.NumObjects;
    props = regionprops(CC, 'Area');
    areas = [props.Area];
    max_area = max(areas);
    total_area = sum(areas);
    
    % Risk calculation
    if max_area > 150
        size_risk = 0.9;
    elseif max_area > 80
        size_risk = 0.6;
    elseif max_area > 35
        size_risk = 0.3;
    else
        size_risk = 0.1;
    end
    
    if num_aneurysms > 2
        multiplicity_risk = 0.8;
    elseif num_aneurysms > 1
        multiplicity_risk = 0.5;
    else
        multiplicity_risk = 0.2;
    end
    
    % Morphology risk from scores
    if ~isempty(aneurysm_scores)
        avg_circularity = mean(aneurysm_scores(:, 2));
        morphology_risk = avg_circularity * 0.8;
    else
        morphology_risk = 0.3;
    end
    
    risk_weights = [0.5, 0.3, 0.2];
    combined_risk = risk_weights(1)*size_risk + ...
                    risk_weights(2)*morphology_risk + ...
                    risk_weights(3)*multiplicity_risk;
    
    if combined_risk > 0.7
        risk_level = 'HIGH';
        recommendation = 'Urgent clinical correlation recommended';
    elseif combined_risk > 0.4
        risk_level = 'MODERATE';
        recommendation = 'Follow-up imaging suggested';
    elseif combined_risk > 0.2
        risk_level = 'LOW';
        recommendation = 'Routine monitoring';
    else
        risk_level = 'MINIMAL';
        recommendation = 'No immediate action required';
    end
    
    risk_assessment.has_aneurysm = true;
    risk_assessment.num_aneurysms = num_aneurysms;
    risk_assessment.max_size_pixels = max_area;
    risk_assessment.total_area_pixels = total_area;
    risk_assessment.risk_score = combined_risk;
    risk_assessment.risk_level = risk_level;
    risk_assessment.recommendation = recommendation;
end

% =========================================================================
% PROCESSING PIPELINE - WITH REGION SPLITTING
% =========================================================================
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('SWI ANEURYSM DETECTION - WITH REGION SPLITTING (12 REGIONS)\n');
fprintf('%s\n', repmat('=', 1, 70));

% Step 1: Skull stripping
fprintf('\nStep 1: Removing skull...\n');
brain_mask = skull_strip_swi(original);
brain_only = original;
brain_only(~brain_mask) = 0;

% Step 2: Create brain interior mask
fprintf('Step 2: Creating brain interior mask...\n');
interior_mask = create_brain_interior_mask(brain_mask, 8);

% Step 3: Wavelet denoising
fprintf('Step 3: Applying wavelet denoising...\n');
brain_denoised = wavelet_denoise_swi(brain_only);
brain_denoised(~brain_mask) = 0;

% Step 4: Global enhancement
fprintf('Step 4: Applying contrast enhancement...\n');
brain_enhanced = adapthisteq(brain_denoised, ...
    'NumTiles', [8 8], ...
    'ClipLimit', 0.02, ...
    'Distribution', 'rayleigh', ...
    'Alpha', 0.5);
brain_enhanced(~brain_mask) = 0;

% =========================================================================
% Step 5: Region-based Aneurysm Detection
% =========================================================================
fprintf('Step 5: Performing region-based aneurysm detection (12 regions)...\n');
[blended_aneurysm_map, region_candidates, region_scores] = ...
    analyze_regions_for_aneurysms(brain_enhanced, interior_mask, 12);

% Step 6: Classify aneurysms
fprintf('Step 6: Classifying aneurysms...\n');
[aneurysm_mask, aneurysm_scores, aneurysm_regions] = ...
    classify_aneurysms(blended_aneurysm_map, interior_mask);

% Step 7: Risk assessment
fprintf('Step 7: Performing risk assessment...\n');
risk_assessment = assess_aneurysm_risk(aneurysm_mask, aneurysm_scores);

% =========================================================================
% CREATE VISUALIZATIONS
% =========================================================================

% 1. ORIGINAL IMAGE
original_display = original;

% 2. SPLIT BRAIN REGIONS (with grid overlay)
regions_display = original;
regions_display = im2double(regions_display);
regions_display = repmat(regions_display, [1,1,3]);

% Get region boundaries
[region_coords, ~, ~] = split_into_focused_regions(interior_mask, 12);

% Draw region boundaries
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

% 3. EXTRACTED BRAIN
brain_display = brain_only;
brain_display = im2double(brain_display);
brain_display = repmat(brain_display, [1,1,3]);
for c = 1:3
    channel = brain_display(:,:,c);
    channel(~brain_mask) = 1;
    brain_display(:,:,c) = channel;
end

% 4. ANEURYSM DETECTION RESULT
result_display = 255 * ones(size(brain_enhanced), 'uint8');
result_display = repmat(result_display, [1,1,3]);
result_display = im2double(result_display);

% Draw aneurysm candidates in yellow
if any(blended_aneurysm_map(:))
    for c = 1:3
        channel = result_display(:,:,c);
        if c == 1 || c == 2  % Yellow
            channel(blended_aneurysm_map) = 1;
        else
            channel(blended_aneurysm_map) = 0;
        end
        result_display(:,:,c) = channel;
    end
end

% Draw final aneurysms in red (overlay)
if any(aneurysm_mask(:))
    for c = 1:3
        channel = result_display(:,:,c);
        if c == 1  % Red
            channel(aneurysm_mask) = 1;
        else
            channel(aneurysm_mask) = 0;
        end
        result_display(:,:,c) = channel;
    end
end

% Background to white
for c = 1:3
    channel = result_display(:,:,c);
    channel(~brain_mask) = 1;
    result_display(:,:,c) = channel;
end

% =========================================================================
% MAIN VISUALIZATION - 4-PANEL PLOT
% =========================================================================
figure('Name', 'SWI Aneurysm Detection - Region-Based Analysis', ...
       'Position', [100, 100, 1400, 1000], ...
       'Color', 'white');

% Panel 1: Original Image
subplot(2,2,1);
imshow(original_display);
title('A. ORIGINAL SWI IMAGE', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

% Panel 2: Split Brain Regions
subplot(2,2,2);
imshow(regions_display);
title('B. SPLIT BRAIN REGIONS (12 Regions)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;
text(0.02, 0.98, '12 analysis regions with overlap', 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, ...
     'BackgroundColor', 'black');

% Panel 3: Extracted Brain
subplot(2,2,3);
imshow(brain_display);
title('C. EXTRACTED BRAIN (Skull Stripped)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

% Panel 4: Aneurysm Detection Result
subplot(2,2,4);
imshow(result_display);
title('D. DETECTED ANEURYSMS', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

if any(aneurysm_mask(:))
    text(0.02, 0.98, sprintf('Aneurysms: %d', risk_assessment.num_aneurysms), ...
         'Units', 'normalized', 'VerticalAlignment', 'top', 'Color', 'white', ...
         'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'red');
else
    text(0.02, 0.98, 'NO ANEURYSMS DETECTED', 'Units', 'normalized', ...
         'VerticalAlignment', 'top', 'Color', 'green', 'FontSize', 12, ...
         'FontWeight', 'bold', 'BackgroundColor', 'white');
end

text(0.02, 0.92, 'Yellow=Candidates, Red=Final', 'Units', 'normalized', ...
     'VerticalAlignment', 'top', 'Color', 'black', 'FontSize', 10, ...
     'BackgroundColor', 'white');

sgtitle('SWI Brain Aneurysm Detection - Region-Based Analysis (12 Regions)', ...
        'FontSize', 16, 'FontWeight', 'bold');

% =========================================================================
% SUPPLEMENTARY FIGURE - Region Analysis Detail
% =========================================================================
figure('Name', 'Region Analysis Detail', ...
       'Position', [150, 150, 1200, 600], ...
       'Color', 'white');

subplot(1,2,1);
% Show region grid
imshow(regions_display);
title('Analysis Regions (12 Non-Overlapping)', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

subplot(1,2,2);
% Show final result
imshow(result_display);
title('Final Aneurysm Detection', 'FontSize', 14, 'FontWeight', 'bold');
axis image off;

% =========================================================================
% STATISTICS REPORT
% =========================================================================
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('ANEURYSM DETECTION RESULTS - 4-PANEL SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 70));

brain_pixels = sum(brain_mask(:));
interior_pixels = sum(interior_mask(:));
edge_pixels = brain_pixels - interior_pixels;

fprintf('\nBRAIN STATISTICS:\n');
fprintf('  Total brain pixels: %d\n', brain_pixels);
fprintf('  Interior brain pixels: %d (%.1f%%)\n', interior_pixels, 100*interior_pixels/brain_pixels);
fprintf('  Edge pixels suppressed: %d (%.1f%%)\n', edge_pixels, 100*edge_pixels/brain_pixels);

fprintf('\nREGION STATISTICS:\n');
fprintf('  Number of analysis regions: 12\n');
fprintf('  Region overlap: 10 pixels\n');
fprintf('  Grid arrangement: 3 rows x 4 columns\n');

fprintf('\nANEURYSM DETECTION:\n');
fprintf('  Aneurysm candidates found: %d\n', sum(blended_aneurysm_map(:)));
fprintf('  Aneurysms classified: %d\n', risk_assessment.num_aneurysms);

if risk_assessment.has_aneurysm
    fprintf('\nRISK ASSESSMENT:\n');
    fprintf('  Risk Level: %s\n', risk_assessment.risk_level);
    fprintf('  Risk Score: %.2f\n', risk_assessment.risk_score);
    fprintf('  Largest aneurysm: %.0f pixels\n', risk_assessment.max_size_pixels);
    fprintf('  Recommendation: %s\n', risk_assessment.recommendation);
end

fprintf('%s\n', repmat('=', 1, 70));
fprintf('\n=== Processing Complete! ===\n');