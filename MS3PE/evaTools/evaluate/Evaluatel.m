function res = Evaluatel(oriResPath, DataSet, w_occ, maxDist)
    % Local pc matlab
    
    fprintf(repmat('-', 1, 99))
    fprintf("\n")
    disp(oriResPath);
    % disp(DataSet);
    fprintf("\n")
    % print(nnnn)
    % yyyyyy

    currentPath = fileparts(mfilename('fullpath'));

    addpath(genpath('/user/xuli/researchfiler/deocc/MS3PE/evaTools/doobscripts'));

    % addpath(genpath('doobscripts'));
    % addAttachedFiles(gcp,["EvaluateOcclusion.m" "Evaluate.m"])

    
    switch DataSet
        case 'PIOD'
            oriGtPath = '/user/xuli/researchfiler/occdata/PIOD/Data';
            testIdsFilename = '/user/xuli/researchfiler/occdata/PIOD/val_doc_2010.txt';

            opt.method_folder = 'TPENet';
            opt.model_name = 'PIOD';
            
        case 'BSDSownership'
            oriGtPath = '/user/xuli/researchfiler/occdata/BSDSownership/testfg/';
            testIdsFilename = '/user/xuli/researchfiler/occdata/BSDSownership/Augmentation/test_ori_iids.lst';

            opt.method_folder = 'TPENet';
            opt.model_name = 'BSDSownership';

        case 'nyuv2_ocpp'
            oriGtPath = '/user/xuli/researchfiler/occdata/NYUv2-OCpp/ob/';
            testIdsFilename = '/user/xuli/researchfiler/occdata/NYUv2-OCpp/test.txt';

            opt.method_folder = 'TPENet';
            opt.model_name = 'NYUV2';
            
        case 'ibims'
            oriGtPath  = '';
            testIdsFilename = '/data/ibims/test_ori_iids.txt'
            
            opt.method_folder = 'TPENet';
            opt.model_name  = 'ibims';  
                        
            
        case 'synocc_test'
            oriGtPath = '/home/xuli/researchfiler/occdata/soccdsf/';
            testIdsFilename = '/home/xuli/researchfiler/deocc/i2opnet/mtorl/datasets/occ_split/synocc_test200.txt';            
            % need to resize image to accelerate
            % current h/2, w/2 and resize method is bicubic (bilinear)
            % one way to resize is after reading gt, another way is after edge nms
            opt.method_folder = 'TPENet';
            opt.model_name = 'syn-occ1080f-test';
            
        case 'synocc_val'
            % resize image to accelerate? 
            % current h/2, w/2 and resize method is bicubic (bilinear)
            % one way to resize is after reading gt, another way is after edge nms
            oriGtPath = '/user/xuli/researchfiler/occdata/synocc/';
            testIdsFilename = '/user/xuli/researchfiler/deocc/i2opnet/mtorl/datasets/occ_split/synocc_fval.txt';

            opt.method_folder = 'TPENet';
            opt.model_name = 'syn-occ1080f-val';            
            
        case 'diode'
            oriGtPath = '/user/xuli/researchfiler/occdata/OB-DIODE/gt/';
            testIdsFilename = '/user/xuli/researchfiler/occdata/DIODE/diode_val50.txt';

            opt.method_folder = 'TPENet';
            opt.model_name = 'ByDIODE';    

        case 'entityseg'
            oriGtPath = '/user/xuli/researchfiler/occdata/OB-EntitySeg/gt/';
            testIdsFilename = '/user/xuli/researchfiler/occdata/EntitySeg/entityseg_val70.txt';
            % disp(oriGtPath);
            % disp(testIdsFilename);
            % oriGtPath = '/user/xuli/researchfiler/occdata/entityseg/gt/';
            % testIdsFilename = '/user/xuli/researchfiler/occdata/EntitySeg/entityseg_val_single.txt';
            disp(oriGtPath);
            disp(testIdsFilename);
            
            opt.method_folder = 'TPENet';
            opt.model_name = 'ByEntitySeg';           

        case 'cmu'
            oriGtPath = '/user/xuli/researchfiler/occdata/CMU/gt/';
            testIdsFilename = '/user/xuli/researchfiler/occdata/CMU/cmu_val30.txt';
            opt.method_folder = 'TPENet';
            opt.model_name = 'CMU';

        case 'nyudmt'
            oriGtPath = '/user/xuli/researchfiler/occdata/nyu_depth_v2/NYUDv2/edge/';
            testIdsFilename = '/user/xuli/researchfiler/occdata/nyu_depth_v2/NYUDv2/gt_sets/val.txt';
            opt.method_folder = 'MoDOT';
            opt.model_name = 'NYUDMT';
            
    end
    
    ImageList = textread(testIdsFilename, '%s');
    opt.validate = 0;
    
    if strcmp(DataSet, 'diode')
        for i = 1:length(ImageList)
            new_img_path = strrep(ImageList{i}, 'imgs/', '');
            new_img_path = strrep(new_img_path, '.png', '');
            ImageList{i} = new_img_path;
        end
    end
    
    if strcmp(DataSet, 'entityseg')
        for i = 1:length(ImageList)
            new_img_path = strrep(ImageList{i}, 'imgs/', '');
            % new_img_path = strrep(new_img_path, '.png', '');
            ImageList{i} = new_img_path;
        end
    end
    
    if strcmp(DataSet, 'nyuv2_ocpp')
        for i = 1:length(ImageList)
	    new_img_path = strrep(ImageList{i}, '.png', '');
	    ImageList{i} = new_img_path;
	end
    end

    if strcmp(DataSet, 'cmu')
        for i = 1:length(ImageList)
	    new_img_path = strrep(ImageList{i}, '.png', '');
	    ImageList{i} = new_img_path;
	end
    end
        
    if opt.validate
        valid_num = 10;
        ImageList = ImageList(1:valid_num);
    end

    tic;
    respath = [oriResPath, '/'];
    evalPath = [oriResPath, '/eval_fig/']; if~exist(evalPath, 'dir') mkdir(evalPath); end

    opt.DataSet = DataSet;
    opt.maxDist = maxDist;
    opt.vis = 1;
    opt.print = 0;
    opt.overwrite = 1;
    opt.visall = 0;
    opt.append = '';
    opt.occ_scale = 1; % set which scale output for occlusion
    opt.w_occ = w_occ;
    if opt.w_occ; opt.append = '_occ'; end
    opt.scale_id = 0;

    if opt.scale_id ~= 0;
        opt.append = [opt.append, '_', num2str(opt.scale_id)];
    end

    opt.outDir = respath;
    opt.resPath = respath;
    opt.gtPath = oriGtPath;
    opt.nthresh = 99; % threshold to calculate precision and recall
    % it set to 33 in DOC for save runtime but 99 in DOOBNet.
    opt.thinpb = 1; % thinpb means performing nms operation before evaluation.
    opt.renormalize = 0;
    opt.fastmode = 0; % see EvaluateSingle.m

    if (~isfield(opt, 'method') || isempty(opt.method)), opt.method = opt.method_folder; end

    fprintf('Starting evaluate dataset %s, model: %s and %s \n', DataSet, opt.method, opt.append);

    EvaluateOcclusion(ImageList, opt);

    if opt.vis
        close all;

        if strfind(opt.append, '_occ');
            app_name = opt.append;

            opt.eval_item_name = 'Boundary';
            opt.append = [app_name, '_e'];
            boundary_res = plot_multi_eval_v2(opt.outDir, opt, opt.method); title('Edge');

            opt.eval_item_name = 'Orientation PR';
            opt.append = [app_name, '_poc'];
            ori_res = plot_multi_eval_v2(opt.outDir, opt, opt.method); title('PRO');

            opt.append = [app_name, '_aoc'];
            ori_accuracy_res = plot_multi_occ_acc_eval(opt.outDir, opt, opt.method);
            res = [boundary_res; ori_res; ori_accuracy_res];

            bins_num = 8;

            for j = 1:bins_num
                opt.eval_item_name = convertStringsToChars(sprintf("%d_%d", bins_num, j));
                opt.append = convertStringsToChars(sprintf("_occ.txt.%d_%d_bins_poc", bins_num, j));
                plot_multi_eval_v2(opt.outDir, opt, opt.method); title('Edge');
            end

        else
            opt.eval_item_name = 'Boundary';
            boundary_res = plot_multi_eval_v2(opt.outDir, opt, opt.method); title('Edge');
            res = [boundary_res; boundary_res; boundary_res];
        end

    end

    toc;
end
