CUDA_VISIBLE_DEVICES=0 python File_Segmentation.py \
    --file_pth "assets/images" \
    --segment_label "human" \
    --mask_save_pth "assets/images_masks"

rm -rf ./img_logs
rm -rf ./result
rm -rf ./results/
