import torch
from ts.torch_handler.base_handler import BaseHandler
from scale_hyperprior_lightning import ScaleHyperpriorLightning
from scale_hyperprior import ScaleHyperprior
import io
import base64
from PIL import Image
from torchvision import transforms

class ScaleHyperpriorHandler(BaseHandler):
    def initialize(self, ctx):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load('weights.zip', map_location=torch.device('cpu'))

    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            
            x = Image.open(io.BytesIO(image)).convert("RGB")
            x = transforms.Resize((512,512))(x)
            # x = transforms.CenterCrop((512,512))(x)
            x = transforms.ToTensor()(x)
            x = x.view(3, 512, 512)
            images.append(x)
        
        return torch.stack(images).to(self.device)

    def inference(self, data, *args, **kwargs):
        # Implement inference logic here
        #input_tensor = torch.tensor(data).to(self.device)
        output = self.model.forward(torch.tensor(data).to(self.device))
        print("OUTPUT")
        print(type(output[0]))
        print(output[0].size())
        return output[0]

    def postprocess(self, data):
        # Implement postprocessing logic here
        x = data.squeeze(0)
        to_pil = transforms.ToPILImage()
        image = to_pil(x)
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()
        return [byte_arr]
