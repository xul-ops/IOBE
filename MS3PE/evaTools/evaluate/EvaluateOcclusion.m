function EvaluateOcclusion(imglist, varargin)
    opt = struct('Evaluate', 'edge', 'resPath', ' ', 'gtPath', ' ', 'nthresh', 30, ...,
    'renormalize', 1, 'w_occ', 0, 'overwrite', 0, 'dataBase', 'PASCAL', 'scale_id', 0);
    
    % addAttachedFiles(gcp,["EvaluateOcclusion.m" "Evaluate.m"])
    
    opt = CatVarargin(opt, varargin);
    append = opt.append;
    % print()

    if exist(fullfile(opt.outDir, ['eval_bdry_thr', append, '.txt']), 'file') & ~opt.overwrite
        return;
    end

    outDir = opt.outDir;
    resPath = opt.resPath;
    gtPath = opt.gtPath;
    opt.dilate = 0; % do not dilate the ground truth for evaluation
    scale_id = opt.scale_id;
    downsampling_factor = 0.5;
    interpolation_method = 'bicubic'; %'bicubic'; % 'bilinear'
    if scale_id == 0; scale_id = 1; end

    if opt.w_occ
        fname = fullfile(opt.outDir, ['eval', append, '_acc.txt']);
    else
        fname = fullfile(opt.outDir, ['eval_bdry', append, '.txt']);
    end

    if ~exist(fname, 'file') | opt.overwrite
        % assert(length(resfile) == length(gtfile));
        n = length(imglist);
        dov = false(1, n);
        params = cell(1, n);

        for ires = 1:length(imglist)
            prFile = fullfile(outDir, [imglist{ires}, '_ev1', append, '.txt']);
            if exist(prFile, 'file') & ~opt.overwrite; continue; end
            dov(ires) = true;

            % get results
            resfile = fullfile(resPath, [imglist{ires}, '.mat']);

            if exist(resfile, 'file')
                edge_maps = load(resfile);
                edge_maps = edge_maps.edge_ori;

                if opt.w_occ
                    res_img = zeros([size(edge_maps.edge), 2], 'single');
                    res_img(:, :, 1) = edge_maps.edge;
                    res_img(:, :, 2) = edge_maps.ori;
                else
                    res_img = zeros([size(edge_maps.edge), 1], 'single');
                    res_img(:, :, 1) = edge_maps.edge;
                end

                if size(edge_maps, 2) >= 3 && opt.w_occ
                    opt.rank_score = res_img(:, :, 1) + edge_maps{scale_id, 3};
                end

            else
                
                edge_resfile = fullfile(resPath, [imglist{ires}, '_ob.png']);
                edge_map = imread(edge_resfile);
                
                % if opt.thinpb && strcmp(opt.DataSet, 'synocc_test')
                    % perform nms operation first
                    % edge_map = edge_nms(edge_map, 0);
                    % edge_map = imresize(edge_map, downsampling_factor, interpolation_method);
                    % edge_map = edge_nms(edge_map, 0)
		% end  
                         
                edge_map = (255 - double(edge_map)) / 255;

                if opt.w_occ
                    res_img = zeros([size(edge_map), 2], 'single');
                    res_img(:, :, 1) = edge_map;

                    occ_resfile = fullfile(resPath, [imglist{ires}, '_oo.png']);
                    occ_map = imread(occ_resfile);
                    occ_map = double(occ_map) * 2 * pi / 255;
                    res_img(:, :, 2) = occ_map;
                else
                    res_img = zeros([size(edge_map), 1], 'single');
                    res_img(:, :, 1) = edge_map;
                end

            end

            % get ground truth
            switch opt.DataSet
                case 'PIOD'
                    gtfile = fullfile(gtPath, [imglist{ires}, '.mat']);
                    bndinfo_pascal = load(gtfile);
                    bndinfo_pascal = bndinfo_pascal.bndinfo_pascal;
                    gt_img = GetEdgeGT(bndinfo_pascal);

                case 'BSDSownership'
                    gtfile = fullfile(gtPath, [imglist{ires}, '.mat']);
                    gtStruct = load(gtfile);
                    gtStruct = gtStruct.gtStruct;
                    gt_img = cat(3, gtStruct.gt_theta{:});

                case 'ibims'
                    gtfile = fullfile(gtPath, [imglist{ires}, '.mat']);
                    gtStruct = load(gtfile);
                    gt_img = gtStruct.gt_edge_theta;  % edge+ori; H,W,(2+2)
                
                case 'nyuv2_ocpp'
                    % gtfile = fullfile(gtPath, [imglist{ires}, '.mat']);
                    % gtStruct = load(gtfile);
                    % gt_img = gtStruct.gtStruct.gt_theta;  % edge+ori; H,W,(2+2)
                    gtfile = fullfile(gtPath, [imglist{ires}, '.png']);
                    edge_gt = imread(gtfile);
                    % edge_gt = rgb2gray(edge_gt);
                    edge_gt_mask = edge_gt > 0;                    
                    [image_height, image_width, num_channels] = size(edge_gt_mask);

                    gt_img = zeros([image_height, image_width , 2], 'single');
                    gt_img(:,:,1) = edge_gt_mask;

                    if opt.w_occ
                        % TO BE FINISHED 
                        c_img = strrep(ImageList{ires}, '/ob/', '/oo/');
                        gtfile_oo = fullfile(gtPath, [c_img, '.png']);
                        ori_gt = imread(gtfile_oo);
                        ori_gt_mask = double(edge_gt) / 255;
                        gt_img(:,:,2) = ori_gt_mask;
                    else
                        gt_img(:,:,2) = edge_gt_mask;
                    end
                              
                    
                case 'synocc_test'

                    gtfile_ob = fullfile(gtPath, [imglist{ires}, '/dis_fOB.png']);
                    edge_gt = imread(gtfile_ob);
                    % edge_gt= imresize(edge_gt, downsampling_factor, interpolation_method);
                    edge_gt = rgb2gray(edge_gt);
                    edge_gt_mask = edge_gt > 0;                    
                    [image_height, image_width, num_channels] = size(edge_gt_mask);

                    gt_img = zeros([image_height, image_width , 2], 'single');
                    gt_img(:,:,1) = edge_gt_mask;

                    if opt.w_occ
                        % TO BE FINISHED 
                        gtfile_oo = fullfile(gtPath, [imglist{ires}, '/dis_fOO.png']);
                        ori_gt = imread(gtfile_oo);
                        ori_gt_mask = double(edge_gt) / 255;
                        gt_img(:,:,2) = ori_gt_mask;
                    else
                        gt_img(:,:,2) = edge_gt_mask;
                    end
                    
                case 'synocc_val'

                    gtfile_ob = fullfile(gtPath, [imglist{ires}, '/dis_fOB.png']);
                    edge_gt = imread(gtfile_ob);
                    edge_gt = rgb2gray(edge_gt);                   
                    % if startsWith(imglist{ires}, '01306')
                    %     edge_gt = edge_gt(1:768, 1:1024);
                    % else
                    %     edge_gt = edge_gt(1080-768+1:end, 1:1024);
                    % end
                    
                    % edge_gt = rgb2gray(edge_gt);
                    edge_gt_mask = edge_gt > 0;                    
                    [image_height, image_width, num_channels] = size(edge_gt);

                    gt_img = zeros([image_height, image_width , 2], 'single');
                    gt_img(:,:,1) = edge_gt_mask;

                    if opt.w_occ
                        % TO BE FINISHED 
                        fprintf('Function code is not complete! ADD IN FUTURE\n');
                        gtfile_oo = fullfile(gtPath, [imglist{ires}, '/dis_fOO.png']);
                        ori_gt = imread(gtfile_oo);
                        ori_gt_mask = double(edge_gt) / 255;
                        gt_img(:,:,2) = ori_gt_mask;
                    else
                        gt_img(:,:,2) = edge_gt_mask;
                    end
                                        
                case 'diode'

                    gtfile_ob = fullfile(gtPath, [imglist{ires}, '.png']);
                    edge_gt = imread(gtfile_ob);
                    edge_gt_mask = edge_gt > 0;                                      
                    [image_height, image_width, num_channels] = size(edge_gt);

                    gt_img = zeros([image_height, image_width ,2], 'single');
                    gt_img(:,:,1) = edge_gt_mask;

                    if opt.w_occ
                        % TO BE FINISHED 
                        fprintf('Function code is not complete and currently dont have DIODE ORI! ADD IN FUTURE\n');
                        gt_img(:,:,2) = edge_gt_mask;
                    else
                        gt_img(:,:,2) = edge_gt_mask;
                        
                    end
                    
                case 'entityseg'

                    gtfile_ob = fullfile(gtPath, [imglist{ires}, '.png']);
                    edge_gt = imread(gtfile_ob);
                    edge_gt = rgb2gray(edge_gt);  
                    edge_gt_mask = edge_gt > 0;                    
                    [image_height, image_width, num_channels] = size(edge_gt);

                    gt_img = zeros([image_height, image_width ,2], 'single');
                    gt_img(:,:,1) = edge_gt_mask;

                    if opt.w_occ
                        % TO BE FINISHED 
                        fprintf('Function code is not complete and currently dont have DIODE ORI! ADD IN FUTURE\n');
                        gt_img(:,:,2) = edge_gt_mask;
                    else
                        gt_img(:,:,2) = edge_gt_mask;
                        
                    end
                    
                case 'cmu'

                    gtfile_ob = fullfile(gtPath, [imglist{ires}, '_fragmentgroundtruth.png']);
                    edge_gt = imread(gtfile_ob);
                    edge_gt = im2gray(edge_gt);  
                    edge_gt_mask = edge_gt > 0;                    
                    [image_height, image_width, num_channels] = size(edge_gt);

                    gt_img = zeros([image_height, image_width ,2], 'single');
                    gt_img(:,:,1) = edge_gt_mask;

                    if opt.w_occ
                        % TO BE FINISHED 
                        fprintf('Function code is not complete and currently dont have DIODE ORI! ADD IN FUTURE\n');
                        gt_img(:,:,2) = edge_gt_mask;
                    else
                        gt_img(:,:,2) = edge_gt_mask;
                        
                    end

                case 'nyudmt'

                    gtfile_ob = fullfile(gtPath, [imglist{ires}, '.png']);
                    edge_gt = imread(gtfile_ob);
                    edge_gt = im2gray(edge_gt);  
                    edge_gt_mask = edge_gt > 0;                    
                    [image_height, image_width, num_channels] = size(edge_gt);

                    gt_img = zeros([image_height, image_width ,2], 'single');
                    gt_img(:,:,1) = edge_gt_mask;

                    if opt.w_occ
                        % TO BE FINISHED 
                        fprintf('Function code is not complete and currently dont have DIODE ORI! ADD IN FUTURE\n');
                        gt_img(:,:,2) = edge_gt_mask;
                    else
                        gt_img(:,:,2) = edge_gt_mask;
                        
                    end
                    
            end


            %         if opt.vis
            %             imagesc(gt_img(:,:,2));
            %             pause;
            %         end

            if ~all(size(res_img(:, :, 1)) - size(gt_img(:, :, 1)) == 0)
                res_img = imresize(res_img, size(gt_img(:, :, 1)), 'bilinear');

                if isfield(opt, 'rank_score')
                    opt.rank_score = imresize(opt.rank_score, size(gt_img(:, :, 1)), 'bilinear');
                end

            end

            params{ires} = {res_img, gt_img, prFile, opt};
            % EvaluateSingle(res_img, gt_img, prFile, opt);
        end

        params = params(dov);
        n = length(params);
        % parforProgress(n);
        % complete_count = zeros(n, 1);
        w = 50;
        
        parfor i = 1:n
            % for i = 1:n
            feval('EvaluateSingle', params{i}{:});
            % feval('EvaluateSingleBins', params{i}{:});
        end

        % parforProgress(0);
    end

    %% collect results
    if strfind(append, '_occ');
        % opt.overwrite = 1;
        collect_eval_bdry_occ(opt.outDir, append, opt.overwrite);
        bins_num = 8;

        for j = 1:bins_num
            str = sprintf("_occ.txt.%d_%d_bins", bins_num, j);
            collect_eval_bdry_occ(opt.outDir, convertStringsToChars(str), opt.overwrite);
        end

        % _occ.txt.occbins
    else
        collect_eval_bdry_v2(opt.outDir, append);
    end

    %% clean up
    % system(sprintf('rm -f %s/*_ev1%s.txt',opt.outDir,append));

end
