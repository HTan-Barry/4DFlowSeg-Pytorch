dataset=testing
data_dir=./Data
h5_file_name1=aorta03trans
mask_list="0.1 0.2 0.4 0.6 0.8 0.9"
for threshold in $mask_list;
do
root=Data/test_mask_$threshold/
mkdir -p $root
python Patch_Creater.py --dataset $dataset --data_dir $data_dir --h5_file_name1 $h5_file_name1 --mask_threshold $threshold --root $root
done