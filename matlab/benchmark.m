function benchmark(inputPattern, startIndex, endIndex, numRounds, outputFolder, varargin)
    if nargin < 5
        fprintf("Usage: benchmark <input pattern> <index 0> <index n> <rounds> <output folder> [<d1> <d2> ... <dN>]\n");
        return;
    end

    numImages = (endIndex - startIndex) + 1;

    firstFile = sprintf(inputPattern, startIndex);
    firstImg = imread(firstFile);
    [rows, cols] = size(firstImg);

    imageStack = zeros(rows, cols, numImages, 'uint8');
    imageStack(:,:,1) = firstImg;

    for i = 1:(numImages-1)
        filename = sprintf(inputPattern, startIndex + i);
        imageStack(:,:,i+1) = imread(filename);
    end

    totalPixels = rows * cols * numImages;
    if ~isempty(varargin)
        geometry = cell2mat(varargin);
        if prod(geometry) ~= totalPixels
            error('Given geometry (%s) does not match total image pixel size (%d).', mat2str(geometry), totalPixels);
        end
        numDims = length(geometry);
    else
        geometry = [rows, cols, numImages];
        numDims = 3;
    end

    filePattern = '%05d.tif';

    if numDims == 1
        reshapedData = imageStack(:);
    else
        reshapedData = reshape(imageStack, geometry);
    end
    gpuImageStackSingle = gpuArray(single(reshapedData));
    gpuImageStack = gpuArray(uint8(reshapedData));
    gpuDev = gpuDevice();

    curShape = repmat(3, 1, numDims);
    if numDims == 1, curShape = [1, 3]; end
    seMean = ones(curShape) / (3^numDims);

    % No ND structuring element for dilation and erosion :P
    seCube = true(3, 3);
    seCross = [0, 1, 0; 1, 1, 1; 0, 1, 0];
    seCubeSep = cell(1, numDims);
    baseValue = ones(1, 3);
    for i = 1:numDims
        seCubeSep{i} = reshape(baseValue, [3, 1]);
    end

    seMeanSep = cell(1, numDims);
    baseValue = ones(1, 3) / 3;
    for i = 1:numDims
        curShape = ones(1, numDims);
        curShape(i) = 3;
        if numDims == 1
            seMeanSep{i} = baseValue(:);
        else
            seMeanSep{i} = reshape(baseValue, curShape);
        end
    end

    thresholdLevel = 128;

    builder = BenchmarkBuilder();

    builder.attach('upload', 'group', 'memory', ...
        @() gpuArray(single(reshapedData)), ...
        @(name) wait(gpuDev));

    builder.attach('download', 'group', 'memory', ...
        @() gather(gpuImageStackSingle), ...
        @(name) wait(gpuDev));

    builder.attach('copy', 'group', 'memory', ...
        @() gpuImageStack + 0, ...
        @(name) save_copy(gpuImageStack, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex));

    builder.attach('threshold', 'group', 'point', ...
        @() (gpuImageStack > thresholdLevel), ...
        @(name) save_threshold(gpuImageStack, thresholdLevel, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex));

    builder.attach('invert', 'group', 'point', ...
        @() imcomplement(gpuImageStack), ...
        @(name) save_invert(gpuImageStack, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex));

    builder.attach('erode-cross', 'single', '', ...
        @() imerode(gpuImageStack, seCross), ...
        @(name) save_erode_cross(gpuImageStack, seCross, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex));

    builder.attach('erode-cube', 'single', '', ...
        @() imerode(gpuImageStack, seCube), ...
        @(name) save_erode_cube(gpuImageStack, seCube, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex));

    builder.attach('split-erode-cube', 'single', '', ...
        @() perform_split_erode_cube(gpuImageStack, seCubeSep, numDims), ...
        @(name) save_split_erode_cube(gpuImageStack, seCubeSep, numDims, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex));

    builder.attach('convolve', 'single', '', ...
        @() convn(gpuImageStackSingle, seMean, 'same'), ...
        @(name) save_convolve(gpuImageStackSingle, seMean, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex));

    builder.attach('split-convolve', 'single', '', ...
        @() perform_split_convolve(gpuImageStackSingle, seMeanSep, numDims), ...
        @(name) save_split_convolve(gpuImageStackSingle, seMeanSep, numDims, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex));

    builder.run(numRounds);
end

function save_copy(gpuImageStack, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex)
    result = reshape(gather(gpuImageStack + 0), [rows, cols, numImages]);
    [~] = mkdir(fullfile(outputFolder, 'copy'));
    outputResultPattern = fullfile(outputFolder, 'copy', filePattern);
    for i = startIndex:endIndex
        outfilename = sprintf(outputResultPattern, i);
        out = result(:,:,i-startIndex+1);
        imwrite(uint8(out), outfilename);
    end
end

function save_threshold(gpuImageStack, thresholdLevel, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex)
    result = reshape(gather(im2uint8(gpuImageStack > thresholdLevel)), [rows, cols, numImages]);
    [~] = mkdir(fullfile(outputFolder, 'threshold'));
    outputResultPattern = fullfile(outputFolder, 'threshold', filePattern);
    for i = startIndex:endIndex
        outfilename = sprintf(outputResultPattern, i);
        out = result(:,:,i-startIndex+1);
        imwrite(uint8(out), outfilename);
    end
end

function save_invert(gpuImageStack, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex)
    result = reshape(gather(imcomplement(gpuImageStack)), [rows, cols, numImages]);
    [~] = mkdir(fullfile(outputFolder, 'invert'));
    outputResultPattern = fullfile(outputFolder, 'invert', filePattern);
    for i = startIndex:endIndex
        outfilename = sprintf(outputResultPattern, i);
        out = result(:,:,i-startIndex+1);
        imwrite(uint8(out), outfilename);
    end
end

function save_erode_cross(gpuImageStack, seCross, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex)
    result = reshape(gather(imerode(gpuImageStack, seCross)), [rows, cols, numImages]);
    [~] = mkdir(fullfile(outputFolder, 'erode-cross'));
    outputResultPattern = fullfile(outputFolder, 'erode-cross', filePattern);
    for i = startIndex:endIndex
        outfilename = sprintf(outputResultPattern, i);
        out = result(:,:,i-startIndex+1);
        imwrite(uint8(out), outfilename);
    end
end

function save_erode_cube(gpuImageStack, seCube, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex)
    result = reshape(gather(imerode(gpuImageStack, seCube)), [rows, cols, numImages]);
    [~] = mkdir(fullfile(outputFolder, 'erode-cube'));
    outputResultPattern = fullfile(outputFolder, 'erode-cube', filePattern);
    for i = startIndex:endIndex
        outfilename = sprintf(outputResultPattern, i);
        out = result(:,:,i-startIndex+1);
        imwrite(uint8(out), outfilename);
    end
end

function save_split_erode_cube(gpuImageStack, seCubeSep, numDims, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex)
    result = perform_split_erode_cube(gpuImageStack, seCubeSep, numDims);
    result = reshape(gather(result), [rows, cols, numImages]);
    [~] = mkdir(fullfile(outputFolder, 'split-erode-cube'));
    outputResultPattern = fullfile(outputFolder, 'split-erode-cube', filePattern);
    for i = startIndex:endIndex
        outfilename = sprintf(outputResultPattern, i);
        out = result(:,:,i-startIndex+1);
        imwrite(uint8(out), outfilename);
    end
end

function save_convolve(gpuImageStackSingle, seMean, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex)
    result = reshape(gather(convn(gpuImageStackSingle, seMean, 'same')), [rows, cols, numImages]);
    [~] = mkdir(fullfile(outputFolder, 'convolve'));
    outputResultPattern = fullfile(outputFolder, 'convolve', filePattern);
    for i = startIndex:endIndex
        outfilename = sprintf(outputResultPattern, i);
        out = result(:,:,i-startIndex+1);
        imwrite(uint8(out), outfilename);
    end
end

function save_split_convolve(gpuImageStackSingle, seMeanSep, numDims, rows, cols, numImages, outputFolder, filePattern, startIndex, endIndex)
    result = perform_split_convolve(gpuImageStackSingle, seMeanSep, numDims);
    result = reshape(gather(result), [rows, cols, numImages]);
    [~] = mkdir(fullfile(outputFolder, 'split-convolve'));
    outputResultPattern = fullfile(outputFolder, 'split-convolve', filePattern);
    for i = startIndex:endIndex
        outfilename = sprintf(outputResultPattern, i);
        out = result(:,:,i-startIndex+1);
        imwrite(uint8(out), outfilename);
    end
end

function result = perform_split_erode_cube(gpuImageStack, seCubeSep, numDims)
    aux = imerode(gpuImageStack, seCubeSep{1});
    for j = 2:numDims
        if (mod(j, 2) == 0)
            result = imerode(aux, seCubeSep{j});
        else
            aux = imerode(result, seCubeSep{j});
        end
    end
    if (mod(numDims, 2) == 1)
        result = aux;
    end
end

function result = perform_split_convolve(gpuImageStack, seMeanSep, numDims)
    aux = convn(gpuImageStack, seMeanSep{1}, 'same');
    for j = 2:numDims
        if (mod(j, 2) == 0)
            result = convn(aux, seMeanSep{j}, 'same');
        else
            aux = convn(result, seMeanSep{j}, 'same');
        end
    end
    if (mod(numDims, 2) == 1)
        result = aux;
    end
end
