from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from typing import Optional, Union, Tuple


def load_model(model_id: str, return_component: Optional[Union[str, None]] = None) -> Union[None, AutoencoderKL, DDIMScheduler, UNet2DConditionModel, Tuple]:
    valid_components = {'vae', 'scheduler', 'unet'}

    if return_component and return_component not in valid_components:
        raise ValueError(
            f"Invalid return_component. Use one of {valid_components}")

    components = []

    if return_component == 'vae' or not return_component:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        vae.eval()
        components.append(vae)

    if return_component == 'scheduler' or not return_component:
        scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler")
        components.append(scheduler)

    if return_component == 'unet' or not return_component:
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        components.append(unet)

    if len(components) == 0:
        return None
    elif len(components) == 1:
        return components[0]
    else:
        return tuple(components)
