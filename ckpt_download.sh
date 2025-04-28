python ckpt_download.py

mkdir ckpt
mv tmp_ckpt/segment_checkpoints/groundingdino_swint_ogc.pth ./ckpt
mv tmp_ckpt/segment_checkpoints/R50_DeAOTL_PRE_YTB_DAV.pth ./ckpt
mv tmp_ckpt/segment_checkpoints/sam_vit_b_01ec64.pth ./ckpt
mv tmp_ckpt/segment_checkpoints/audio_mdl.pth ./ast_master/pretrained_models

rm -rf ./tmp_ckpt
