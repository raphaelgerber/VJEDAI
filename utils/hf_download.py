from huggingface_hub import hf_hub_download
import os
import shutil

hf_token = os.environ.get("MY_HF_TOKEN")

out_path = './v1.pth'

cached = hf_hub_download(
    repo_id='kalandarX/jdepth',
    filename='large/v1.pth',
    token=hf_token,
)

shutil.copyfile(cached, out_path)
