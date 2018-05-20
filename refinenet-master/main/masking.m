function masking()
filename = '000000020571';
mask = imread(strcat('D:\TUHH\Arbeit\refinenet-master\refinenet-master\cache_data\test_examples_coco\400epoc_result_20180503195732_predict_custom_data\predict_result_4\predict_result_mask\',filename,'.png'));
%mask = imread(strcat('D:\TUHH\Arbeit\refinenet-master\refinenet-master\cache_data\test_examples_voc2012\result_20180430230338_predict_custom_data\predict_result_3\predict_result_mask\',filename,'.png'));
%label = imread(strcat('D:\TUHH\Arbeit\refinenet-master\refinenet-master\datasets\cocostuff_ship_bckg\my_class_idxes_mask\img_idx_3026_'+filename+'.png'));
color_mask = mask == 1;
rgb_img = imread(strcat('D:\TUHH\Arbeit\refinenet-master\refinenet-master\datasets\cocostuff_ship_bckg\JPG_Images\',filename,'.jpg'));
g = rgb_img(:,:,2);
r = rgb_img(:,:,1);
r(color_mask) = 0;
g(color_mask) = 0;
masked_img = cat(3,r,g,rgb_img(:,:,3));
imshow(masked_img);