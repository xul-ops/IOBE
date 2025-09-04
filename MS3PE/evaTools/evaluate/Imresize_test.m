function Imresize_test(resImgDir, outputFolder)
    % Try to speed up the process by resizing
    addpath(genpath('/home/xuli/researchfiler/deocc/MS3PE/evaTools/doobscripts'));
    
    original_image = imread(resImgDir);
    
    edge_prob = (255 - double(original_image)) / 255;  % [0~1]
    edge_prob_thin = edge_nms(edge_prob, 0);
    edge_vis = mat2gray(1-edge_prob_thin);
    nms_path  = fullfile(outputFolder, 'nms.png');  
    % imwrite(edge_vis, nms_path);

    downsampling_factor = 0.5;
    interpolation_methods = {'bilinear', 'bicubic'};  % , 'lanczos'

    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end


    for method_idx = 1:length(interpolation_methods)
        method = interpolation_methods{method_idx};
        
        % Start timing
        tic;
        
        % Resize the image with the current method
        resized_image = imresize(original_image, downsampling_factor, method);
        
        % End timing
        elapsed_time = toc;
        
        % Save the resized image with method and time in the filename
        output_filename = sprintf('%s/%s_%.2f.png', outputFolder, method, downsampling_factor);
        imwrite(resized_image, output_filename);
        
        % Display and save the time taken
        fprintf('Method: %s, Downsampling Factor: %.2f, Time: %.4f seconds\n', method, downsampling_factor, elapsed_time);

    end

end


