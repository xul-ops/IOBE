function MatlabEdgeNMS(resRootDir, ImageList)
    % script to do edge detection nms 
    % dataset = 'nyuv2';
    addpath(genpath('/home/xuli/researchfiler/deocc/MS3PE/evaTools/doobscripts'));

    % ImageList = textread(testIdsFilename, '%s');
    
    % for ires = 1:length(ImageList)
    parfor ires = 1:length(ImageList)
        prob_in_path  = fullfile(resRootDir, [ImageList{ires}, '.jpg']);  
        prob_out_path = fullfile(resRootDir, [ImageList{ires}, '_nms.png']);  
        
        % if endsWith(prob_in_path, '_ob.png')
        % disp(prob_in_path);
        % else
        
        % read imgs
        edge_prob = imread(prob_in_path);  % uint8
        % edge_prob = single(edge_prob) / 255;  % [0~1]
        edge_prob = (255 - double(edge_prob)) / 255;  % [0~1]
        
        % do nms
        edge_prob_thin = edge_nms(edge_prob, 0);
        
        % output imgs
        % edge_prob_thin = cast(edge_prob_thin * 255., 'uint8');
        % imwrite(edge_prob_thin, prob_out_path);
        imwrite(mat2gray(1-edge_prob_thin), prob_out_path);
        % end
end

end


