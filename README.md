Auto-MBW for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) loosely based on [sdweb-auto-MBW](https://github.com/Xerxemi/sdweb-auto-MBW)

### Purpose
This node "advanced > auto merge block weighted" takes two models, merges individual blocks together at various ratios, and automatically rates each merge, keeping the ratio with the highest score. The resulting model will contain the text encoder and VAE sent to the node, without modification. Whether this is a good idea or not is anyone's guess. In practice this makes models that make images the classifier says are good.

### Settings
- Prompt: to generate sample images to be rated
- Sample Count: number of samples per ratio per block to generate
- Search Depth: number of branches to take while choosing ratios to test
- Classifier: model used to rate images

### Search Depth
To calculate ratios to test, the node branches out from powers of 0.5
- A depth of 2 will examine 0.0, 0.5, 1.0
- A depth of 4 will examine 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
- A depth of 6 will examine 33 different ratios

### Classifier
Currently this only supports [cafeai](https://huggingface.co/cafeai) "aesthetic" and "waifu" models.

### Notes
- --highvram flag recommended - both models will be kept in VRAM and the process is much faster
- many hardcoded settings are arbitrary - such as the sampler and block processing order
- generated images are not saved
- the final model is saved in the models/checkpoints directory with a timestamped name
