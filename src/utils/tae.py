from src.utils.demo_utils.taehv import TAEHV
import urllib.request
import os
import torch

class DotDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

class TAEHVDiffusersWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float16
        self.tae_checkpoint_path = "checkpoints/taew2_1.pth"

        self.download_tae()
        self.taehv = TAEHV(checkpoint_path=self.tae_checkpoint_path).to(self.dtype)
        self.config = DotDict(scaling_factor=1.0)
    
    def download_tae(self):
        if not os.path.exists(self.tae_checkpoint_path):
            print(f"taew2_1.pth not found in checkpoints folder {self.tae_checkpoint_path}. Downloading...")
            os.makedirs("checkpoints", exist_ok=True)
            download_url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
            try:
                urllib.request.urlretrieve(download_url, self.tae_checkpoint_path)
                print(f"Successfully downloaded taew2_1.pth to {self.tae_checkpoint_path}")
            except Exception as e:
                print(f"Failed to download taew2_1.pth: {e}")
                raise

    def encode_to_latent(self, x, mem=None):
        # x: NTCHW value [0,1]
        # Returns (latent, mem) tuple
        return self.taehv.encode_video(x, parallel=False, show_progress_bar=False, mem=mem)

    def decode_to_pixel(self, latents, use_cache=False):
        return self.taehv.decode_video(latents, parallel=False).mul_(2).sub_(1)

    @property
    def model(self):
        # For compatibility with .clear_cache()
        class Dummy:
            def clear_cache(self_inner): pass
        return Dummy()
