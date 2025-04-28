from huggingface_hub import snapshot_download

snapshot_download(repo_id="gulucaptain/DynamiCtrl", allow_patterns="segment_checkpoints/*", local_dir="./tmp_ckpt")