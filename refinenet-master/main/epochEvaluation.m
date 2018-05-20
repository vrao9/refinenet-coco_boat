function epochEvaluation()
files  = epoch_names(2);
path = 'D:\TUHH\Arbeit\result\model_20180511162038\model_cache';
addpath('D:\TUHH\Arbeit\refinenet-master\refinenet-master\main\my_utils');
dir_matConvNet='D:\TUHH\Arbeit\refinenet-master\refinenet-master\libs\matconvnet\matlab';
run(fullfile(dir_matConvNet, 'vl_setupnn.m'));
run_config=[];
ds_config=[];

% run_config.use_gpu=true;
run_config.use_gpu=false;
run_config.gpu_idx=1;

ds_name_subfix = 'coco';
result_name=['result_' datestr(now, 'YYYYmmDDHHMMSS') '_predict_custom_data'];

%%%%%%%%%%%%%%multiple_scales%%%%%%%%%%%%%%%%%%
%pred_scales{1} = [0.8];
%pred_scales{2} = [1.0];
pred_scales{1} = [1 1.2];
pred_scales{2} = [0.8 1.0 1.2];
%pred_scales{5} = [0.6 0.8 1];
%pred_scales{6} = [0.6 0.8 1.0 1.2];

%%%%%%%%%%%%%displaying_scales%%%%%%%%%%%%%%%%
disp('Scales_considered_for_this_evaluation');
for k=1:length(pred_scales)
    disp(pred_scales{k});
end


for k=1:length(pred_scales)
%%%%%%%%%%%%%%diary setting %%%%%%%%%%%%%%%%%%%
    diary_dir=fullfile('D:/TUHH/Arbeit/refinenet-master/refinenet-master/cache_data', ['test_examples_' ds_name_subfix], result_name,['prediction_scales_' int2str(k)]);
    mkdir_notexist(diary_dir);
    diary(fullfile(diary_dir, 'output.txt'));
    diary on

    
    disp('Prediction_scales:');
    prediction_scales = pred_scales{k};
    disp(prediction_scales);
    epoch_no = 0;
    for file = files
        % Do some stuff
        rng('shuffle');
        epoch_no = epoch_no + 5;
        %-------------------------------------------------------------------------------------------------------------------------
        % settings for using trained model:



        % result dir:
        %result_dir=fullfile('D:/TUHH/Arbeit/refinenet-master/refinenet-master/cache_data', ['test_examples_' ds_name_subfix], result_name,['epoch_no_' int2str(epoch_no)]);
        result_dir=fullfile('D:/TUHH/Arbeit/refinenet-master/refinenet-master/cache_data', ['test_examples_' ds_name_subfix], result_name,['prediction_scales_' int2str(k) ] ,['epoch_no_' int2str(epoch_no)]);
        % using a trained model:
        % run_config.trained_model_path='D:/TUHH/Arbeit/refinenet-master/refinenet-master//model_trained/refinenet_res101_voc2012.mat'; % resnet101 based refinenet
        % run_config.trained_model_path='D:/TUHH/Arbeit/refinenet-master/refinenet-master/model_trained/refinenet_res152_voc2012.mat'; % resnet152 based refinenet
        mat_file = strcat('net-config-','epoch-',int2str(epoch_no),'.mat');
        run_config.trained_model_path= strcat(path,'\',file,'\',mat_file);% path to epochs
        run_config.trained_model_path = run_config.trained_model_path{1};
        % provide class_info for the trained model:
        % ds_config.class_info=gen_class_info_voc();
        ds_config.class_info=gen_class_info_coco_ship_bckg();

        % control the size of input images to avoid excessively small or large images
        run_config.input_img_short_edge_min=450;
        run_config.input_img_short_edge_max=800;
        %-------------------------------------------------------------------------------------------------------------------------
        % settings for multi-scale prediction, you can consider 3 scales or 5 scales:

        % prediction_scales=[0.4 0.6 0.8 1 1.2]; % 5 scales
        % prediction_scales=[0.6 0.8 1]; % 3 scales
        % prediction_scales=[1]; % or only use 1 scale
        %-------------------------------------------------------------------------------------------------------------------------
        result_evaluate_param=[];
        result_evaluate_param.gt_mask_dir = 'D:\TUHH\Arbeit\Data\VOCdevkit\voc2012_trainval\SegmentationClass_boat+bckg_trainval';
        ds_config.result_evaluate_param=result_evaluate_param;

        run_config.prediction_scales=prediction_scales;
        run_config.model_name=result_name;
        run_config.root_cache_dir=result_dir;

        do_predict_and_evaluate(run_config, ds_config);

    end
    
end
diary off
end

function do_predict_and_evaluate(run_config, ds_config)

run_config.gen_net_opts_fn=@gen_net_opts_model_type1;

run_config.run_evaonly=true;
ds_config.use_custom_data=false;
ds_config.use_dummy_gt=false;
%
gen_ds_info_fn=@my_gen_ds_info_coco;
ds_config.gen_ds_info_fn=gen_ds_info_fn;
ds_name='voc2012_trainval_boat+bckg';
ds_config.ds_info_cache_dir=fullfile('D:/TUHH/Arbeit/refinenet-master/refinenet-master/datasets', ds_name);
%
run_config.use_dummy_gt=ds_config.use_dummy_gt;

%to change the dataset info, replace the name
ds_config.ds_name='voc2012_trainval_boat+bckg';
mkdir_notexist(run_config.root_cache_dir);


%diary_dir=run_config.root_cache_dir;
%mkdir_notexist(diary_dir);
%diary(fullfile(diary_dir, 'output.txt'));
%diary on


run_dir_name=fileparts(mfilename('fullpath'));
[~, run_dir_name]=fileparts(run_dir_name);
run_config.run_dir_name=run_dir_name;
run_config.run_file_name=mfilename();

ds_info=gen_dataset_info(ds_config);
ds_info.test_idxes = ds_info.test_idxes(1:2);
my_diary_flush();

if run_config.use_gpu
	gpu_num=gpuDeviceCount;
	if gpu_num>=1
		gpuDevice(run_config.gpu_idx);
    else
        error('no gpu found!');
	end
end



train_opts=run_config.gen_net_opts_fn(run_config, ds_info.class_info);

imdb=my_gen_imdb(train_opts, ds_info);
data_norm_info=[];
data_norm_info.image_mean=128;
imdb.ref.data_norm_info=data_norm_info;

[net_config, net_exp_info]=prepare_running_model(train_opts);

prediction_scales=run_config.prediction_scales;
scale_num=length(prediction_scales);
predict_result_dirs=cell(scale_num, 1);

for s_idx=1:scale_num

	one_scale=prediction_scales(s_idx);
	one_result_dir=fullfile(run_config.root_cache_dir, sprintf('predict_result_%d', s_idx));
	predict_result_dirs{s_idx}=one_result_dir;

	fprintf('\n\n--------------------------------------------------\n\n');
	fprintf('conduct prediction using image scale: %1.2f  (current scale / total scales: %d/%d) \n', one_scale, s_idx, scale_num);

	train_opts.root_cache_dir=one_result_dir;
	train_opts.input_img_scale=one_scale;
	train_opts.eva_param=update_eva_param_mscale(train_opts.eva_param, train_opts);

	my_net_tool(train_opts, imdb, net_config, net_exp_info);

	fprintf('\n\n--------------------------------------------------\n\n');
	disp('results are saved in:');
	disp(train_opts.root_cache_dir);

	my_diary_flush();

end


if length(predict_result_dirs)>1
    
    fprintf('\n\n--------------------------------------------------\n\n');
    disp('fusing multiscale predictions');

    fuse_param=[];
    fuse_param.predict_result_dirs=predict_result_dirs;
    fuse_param.fuse_result_dir=fullfile(run_config.root_cache_dir, 'predict_result_final_fused');
    fuse_param.cache_fused_score_map=true;
    fuse_multiscale_results(fuse_param, ds_config.class_info);

    fprintf('\n\n--------------------------------------------------\n\n');
    disp('final fused results are saved in:');
    disp(fuse_param.fuse_result_dir);

    my_diary_flush();

    final_prediction_dir=fuse_param.fuse_result_dir;
    
else
    
    final_prediction_dir=predict_result_dirs{1};
end


% evaluate the final fused result, if ground-truth is provided
result_evaluate_param=ds_config.result_evaluate_param;
if ~isempty(result_evaluate_param) && ~isempty(result_evaluate_param.gt_mask_dir)
	result_evaluate_param.predict_result_dir=fullfile(final_prediction_dir, 'predict_result_mask');
	fprintf('\n\n--------------------------------------------------\n\n');
	disp('perform evaluation of the predicted masks');	
	result_info=evaluate_predict_results(result_evaluate_param, ds_config.class_info);
    disp_evaluate_result(result_info, ds_config.class_info);
    
    eva_result_cached_file=fullfile(final_prediction_dir, 'eva_result_info.mat');
    fprintf('saving evaluation result to: %s\n', eva_result_cached_file);
    save(eva_result_cached_file, 'result_info');
end


my_diary_flush();
end

function s = epoch_names(epochs_no)
%s = {epoch_no};
for c = 1:epochs_no
    s{c} = strcat('epoch_',int2str(5*c));
end
end