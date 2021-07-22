model_name=4DFlowNetV1
epoch=402
data_root=./Data
checkpoint=./checkpoints/${model_name}/epoch${epoch}.pt
mask_list="0.1"
do
data_dir=$data_root/test_mask_$threshold/
csv_dir=./log/${model_name}_epoch_${epoch}_mask_$threshold
img_dir=./result/${model_name}_epoch_${epoch}
echo $data_dir
echo $csv_dir

python test_utils.py --data_dir $data_dir --csv_dir $csv_dir --checkpoint $checkpoint --csv_dir $csv_dir --img_dir $img_dir --save_image True
done