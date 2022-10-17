import cv2
import tempfile
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import imwrite
from cog import BaseModel, BasePredictor, Input, Path
from realesrgan import RealESRGANer
from typing import List

from vqfr.demo_util import VQFR_Demo


class ModelOutput(BaseModel):
    image: Path


class Predictor(BasePredictor):

    def setup(self):
        # bg_upsampler = "realesrgan"
        bg_tile = 400
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='RealESRGAN_x2plus.pth',
            model=model,
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )  # need to set False in CPU mode

        arch = 'v2'
        model_name = 'VQFR_v2'
        model_path = f'experiments/pretrained_models/{model_name}.pth'

        upscale = 2
        self.restorer = VQFR_Demo(model_path=model_path, upscale=upscale, arch=arch, bg_upsampler=bg_upsampler)

    def predict(
            self,
            image: Path = Input(description='Input image. Output restored faces and whole image.', ),
            aligned: bool = Input(
                default=False,
                description='Input are aligned faces.',
            ),
    ) -> List[ModelOutput]:

        only_center_face = False

        input_img = cv2.imread(str(image), cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            input_img,
            fidelity_ratio=0,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=True,
        )

        model_output = []

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            # save restored face
            out_path = Path(tempfile.mkdtemp()) / f'output_{idx}.png'
            imwrite(restored_face, str(out_path))
            model_output.append(ModelOutput(image=out_path))

        # save restored img
        if restored_img is not None:
            out_path = Path(tempfile.mkdtemp()) / 'output.png'
            imwrite(restored_img, str(out_path))
            model_output.append(ModelOutput(image=out_path))
        return model_output
