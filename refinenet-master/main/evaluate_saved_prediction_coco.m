

% example code for evaluation of saved prediction masks, e.g., producing IoU scores

function result_info=evaluate_saved_prediction_coco()

addpath('D:\TUHH\Arbeit\refinenet-master\refinenet-master\main\my_utils');

% provide class info, here's an example for VOC dataset.
class_info=gen_class_info_coco_ship_bckg();

% replace by your prediction mask dir:
predict_result_dir='D:\TUHH\Arbeit\refinenet-master\refinenet-master\cache_data\test_examples_coco\result_20180430191546_predict_custom_data\predict_result_1\predict_result_mask';
% predict_result_dir='../cache_data/test_examples_voc/result_20180212182355_evaonly_custom_data_valset_5scales/predict_result_3/predict_result_mask';

% replace by your groundtruth mask dir:
gt_mask_dir='D:\TUHH\Arbeit\refinenet-master\refinenet-master\datasets\cocostuff_ship_bckg\SegmentationClass';

result_evaluate_param=[];
result_evaluate_param.predict_result_dir=predict_result_dir;
result_evaluate_param.gt_mask_dir=gt_mask_dir;

result_cached_filename='eva_result_info.mat';
result_cached_dir=fileparts(result_evaluate_param.predict_result_dir);
result_cached_file=fullfile(result_cached_dir, result_cached_filename);

diary_dir=result_cached_dir;
mkdir_notexist(diary_dir);
diary(fullfile(diary_dir, 'output.txt'));
diary on

result_info=evaluate_predict_results(result_evaluate_param, class_info);
disp_evaluate_result(result_info, class_info);

fprintf('saving evaluation result to: %s\n', result_cached_file);
save(result_cached_file, 'result_info');

my_diary_flush();
diary off

end

