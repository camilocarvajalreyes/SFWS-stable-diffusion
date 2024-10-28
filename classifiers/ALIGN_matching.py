"""
Concept detector that compares the embedding of a concept to that of a "general image".
Uses the multimodal model ALIGN: https://huggingface.co/docs/transformers/en/model_doc/align
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from transformers import AlignProcessor, AlignModel

device = f"cuda:{torch.cuda.device_count()-1}" if torch.cuda.is_available() else "cpu"


class AlignMatching(nn.Module):
    def __init__(self, device=device, images_kwargs=None):
        super(AlignMatching, self).__init__()
        self.device = device
        self.processor = AlignProcessor.from_pretrained("kakaobrain/align-base",
                                                        images_kwargs=images_kwargs,
                                                        device_map=self.device)
        self.model = AlignModel.from_pretrained("kakaobrain/align-base",
                                                device_map=self.device)
        self.base_concept = None
        self.target_concept = None
    
    def load_concept_embeddings(self, concepts):
        # WARNING: currently only optimised to receive a single concept
        if isinstance(concepts,str):
            concepts = [concepts]
        self.target_concept = concepts[0]
        if self.base_concept is not None:
            self.process_text()

    def load_base_embedding(self, concepts):
        if isinstance(concepts,str):
            concepts = [concepts]
        self.base_concept = concepts[0]
        if self.target_concept is not None:
            self.process_text()

    def process_text(self):
        text_input = [self.target_concept,self.base_concept]
        self.processed_text = self.processor(text=text_input, return_tensors="pt").to(self.device)
    
    def preprocess(self,x_input:torch.Tensor):
        """Preprocess a tensor image
        Arguments
            x_input: torch.Tensor
                input tensor of size (b, c=3, h, w) with a batch of images to process
            
        Returns
            x_out = torch.Tensor of size (b, c=3, 224, 224)
        """
        x_out = F.resize(x_input, (289, 289), interpolation=F.InterpolationMode.BICUBIC)
        x_min, x_max = x_out.min(), x_out.max()
        x_out = (x_out - x_min) / (x_max - x_min)
        return x_out

    def forward(self, img_input):
        """
        Returns a set of cosine similarities given of the input, with respect to a set of CLIP embeddings.

        Arguments:
            img_input: torch.Tensor
                input tensor of size (b, c=3, h, w) with a batch of images to process

        Returns:
            probs: torch.Tensor
                probability for target concept
        """
        img_input = img_input.to(self.device).float()

        outputs = self.model(**self.processed_text,pixel_values=img_input)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        probs = probs[:,0].unsqueeze(1)

        return probs
