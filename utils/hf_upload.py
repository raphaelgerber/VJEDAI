from huggingface_hub import HfApi
import os

hf_token = os.environ.get("MY_HF_TOKEN")
api = HfApi(token=hf_token)

ckpt_path = '/work/scratch/msayfiddinov/checkpoints/jdepth_mse/best_si_mse.pth'

api.upload_file(
    path_or_fileobj=ckpt_path,
    path_in_repo='large/v1.1_mse.pth',
    repo_id='kalandarX/jdepth',
    repo_type='model',
)
