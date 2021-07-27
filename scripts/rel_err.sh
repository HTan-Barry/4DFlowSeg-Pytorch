model_name=4DFlowNetV1
epoch=402
data_root=./Data

# mask_list="0.1 0.2 0.4 0.6 0.8 0.9"
mask_list="0.1"
for threshold in $mask_list;
do
data_dir=$data_root/test_mask_$threshold/
csv_dir=./log/${model_name}_epoch_${epoch}_mask_$threshold
img_dir=./result/${model_name}_epoch_${epoch}
echo $data_dir
echo $csv_dir

python rel_error_utils.py --data_dir $data_dir --csv_dir $csv_dir --csv_dir $csv_dir
done

