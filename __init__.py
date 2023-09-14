import importlib
import math
import pathlib
import sys
import warnings

import numpy as np
from PIL import Image
import torch
import tqdm

import folder_paths
import model_management
import nodes

sys.path.append(str(pathlib.Path(__file__).parent))
import classifiers
from nodes_model_merging import CheckpointSave

BLOCK_ORDER = [12, 11, 13, 10, 14, 9, 15, 8, 16, 7, 17, 6, 18,
               5, 19, 4, 20, 3, 21, 2, 22, 1, 23, 0, 24]

class AutoMBW(CheckpointSave):
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "masterpiece girl"
                }),
                "negative": ("STRING", {
                    "multiline": True,
                    "default": "worst quality"
                }),
                "search_depth": ("INT", {"default": 4, "min": 2}),
                "sample_count": ("INT", {"default": 1, "min": 1}),
                "classifier": (classifiers.__all__,),
                "filename_prefix": ("STRING", { "multiline": False, "default": "ambw" }),
            }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "ambw"
    CATEGORY = "advanced"

    @torch.no_grad()
    def merge(self, block, ratio):
        sd1 = self.model1.model.state_dict()
        sd2 = self.model2.model.state_dict()

        self.blocks_backup = {}
        for key in self.blocks[block]:
            self.blocks_backup[key] = sd1[key].clone()
            sd1[key].copy_(sd1[key] * (1 - ratio) + sd2[key] * ratio)

    @torch.no_grad()
    def unmerge(self):
        sd1 = self.model1.model.state_dict()

        for key in self.blocks_backup:
            sd1[key].copy_(self.blocks_backup[key])

    def rate_model(self):
        rating = 0
        for i in range(self.sample_count):
            latent = nodes.common_ksampler(
                self.model1, i, 20, 7.0, "ddim", "normal", self.prompt,
                self.negative, {"samples": torch.zeros([1, 4, 64, 64])},
                denoise=1.0)
            decoded = self.vae.decode(latent[0]["samples"])
            image = Image.fromarray(
                np.clip(255. * decoded.cpu().numpy().squeeze(),
                        0, 255).astype(np.uint8))
            with warnings.catch_warnings():
                # several possible transformers nags
                warnings.filterwarnings('ignore')
                rating += self.classifier(image)
        return rating

    def search(self, block, current, start, depth, maximum):
        if depth > self.search_depth or current > 1 or current < 0:
            return maximum

        self.merge(block, current)
        score = self.rate_model()
        self.unmerge()
        if score > maximum[1]:
            maximum = (current, score)

        step = math.pow(start, depth)
        for test_step in (-step, step):
            score = self.search(block, current + test_step, start, depth + 1,
                               maximum)
            if score[1] > maximum[1]:
                maximum = score
        return maximum

    def ambw(self, model1, model2, clip, vae, prompt, negative, search_depth,
             sample_count, classifier, filename_prefix):
        # python setup
        self.output_dir = folder_paths.get_output_directory()
        self.model1 = model1
        self.model2 = model2
        self.vae = vae
        self.prompt = [[clip.encode(prompt), {}]]
        self.negative = [[clip.encode(negative), {}]]
        self.search_depth = search_depth
        self.sample_count = sample_count
        self.classifier = importlib.import_module(
            "." + classifier, "classifiers").score

        # model setup
        if model_management.vram_state == model_management.VRAMState.HIGH_VRAM:
            model1.model.to(model_management.get_torch_device())
            model2.model.to(model_management.get_torch_device())

        self.ratios = [None] * 25
        self.blocks = [None] * 25
        sd1 = model1.model.state_dict()
        self.blocks[12] = [key for key in sd1 if "middle_block" in key]
        for index in range(12):
            self.blocks[index] = \
                [key for key in sd1 if f"input_blocks.{index}." in key]
            self.blocks[index + 13] = \
                [key for key in sd1 if f"output_blocks.{index}." in key]

        def tqdm_steps(depth):
            if depth < 3:
                return 3
            return math.pow(2, depth - 2) + tqdm_steps(depth - 1)

        for block in tqdm.tqdm(BLOCK_ORDER, desc='automerge', unit='block',
                               position=int(tqdm_steps(
                                   search_depth) * sample_count)):
            self.ratios[block] = self.search(block, 0.5, 0.5, 1, (0.5, 0))[0]
            self.merge(block, self.ratios[block])
            print(self.ratios)

        self.save(self.model1, clip, vae, filename_prefix)

        return ()


NODE_CLASS_MAPPINGS = {
    "Auto Merge Block Weighted": AutoMBW
}
