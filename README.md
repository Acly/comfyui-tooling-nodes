# ComfyUI Nodes for External Tooling

Provides nodes and API geared towards using ComfyUI as a backend for external tools.

## Nodes for sending and receiving images

ComfyUI exchanges images via the filesystem. This requires a
multi-step process (upload images, prompt, download images), is rather
inefficient, and invites a whole class of potential issues. It's also unclear
at which point those images will get cleaned up if ComfyUI is used
via external tools.

### Load Image (Base64)

Loads an image from a PNG embedded into the prompt as base64 string.
* Inputs: base64 encoded binary data of a PNG image
* Outputs: image (RGB) and mask (alpha) if present

### Load Mask (Base64)

Loads a mask (single channel) from a PNG embedded into the prompt as base64 string.
* Inputs: base64 encoded binary data of a PNG image
* Outputs: the first channel of the image as mask

### Send Image (WebSocket)

Sends an output image over the client WebSocket connection as PNG binary data.
* Inputs: the image (RGB or RGBA)

This will first send one binary message for each image in the batch via WebSocket:
```
12<PNG-data>
```
That is two 32-bit integers (big endian) with values 1 and 2 followed by the PNG binary data. There is also a JSON message afterwards:
```
{'type': 'executed', 'data': {'node': '<node ID>', 'output': {'images': [{'source': 'websocket', 'content-type': 'image/png', 'type': 'output'}, ...]}, 'prompt_id': '<prompt ID>}}
```

## Nodes for working on regions

These nodes implement attention masking for arbitrary number of image regions. Text prompts only apply to the masked area.
In contrast to condition masking, this method is less "forceful", but leads to more natural image compositions.

![Regions Attention Mask](workflows/region_attention_mask.png)
[Workflow: region_attention_mask.json](workflows/region_attention_mask.json)

### Background Region

This node starts a list of regions. It takes a prompt, but no mask. The prompt is assigned to all image areas which are _not_
covered by another region mask in the list.

### Define Region

Appends a new region to a region list (or starts a new list). Takes a prompt, and mask which defines the area in the image
the prompt will apply to. Masks must be the same size as the image _or_ the latent (which is factor 8 smaller).

### List Region Masks

This node takes a list of regions and outputs all their masks. It can be useful for inspection, debugging or to reuse the
computed background mask.

### Regions Attention Mask

Patches the model to use the provided list of regions. This replaces the positive text conditioning which is provided
to the sampler. It's still possible to pass ControlNet and other conditioning to the sampler.


### Apply Mask to Image

Copies a mask into the alpha channel of an image.
* Inputs: image and mask
* Outputs: RGBA image with mask used as transparency

## API for model inspection

There are various types of models that can be loaded as checkpoint, LoRA, ControlNet, etc. which cannot be used interchangeably. The following API helps to categorize and filter them.

### /api/etn/model_info

Lists available models with additional classification info.
* Paramters: _none_
* Output: list of model files
    ```
    {
        "checkpoint_file.safetensors": {
            "base_model": "sd15"|"sd20"|"sd21"|"sd3"|"sdxl"|"ssd1b"|"svd"|"cascade-b"|"cascade-c",
            "is_inpaint": true|false,
            "is_refiner": true|false
        },
        ...
    }
    ```
    The entry is `{"base_model": "unknown"}` for models which are not in safetensors format or do not match any of the known base models.

_Note: currently only supports checkpoints. May add other models in the future._

## Installation

Download the repository and unpack into the `custom_nodes` folder in the ComfyUI installation directory.

Or clone via GIT, starting from ComfyUI installation directory:
```
cd custom_nodes
git clone https://github.com/Acly/comfyui-tooling-nodes.git
```

Restart ComfyUI and the nodes are functional.
