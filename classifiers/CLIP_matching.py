"""
General CLIP-based concept detector that compares the embedding of a concept to that of a "general image".
"""

import torch
import torch.nn as nn
import clip
from PIL import Image
import numpy as np
from einops import rearrange

import torchvision.transforms.functional as F
from torchvision import transforms


device = f"cuda:{torch.cuda.device_count()-1}" if torch.cuda.is_available() else "cpu"


class ConceptMatching(nn.Module):
    def __init__(self, model_name, device=device):
        super(ConceptMatching, self).__init__()
        self.device = device
        self.clip_model, self.preprocess_clip = clip.load(model_name,device=self.device)
        self.base_embeds = None
        self.concept_embeds = None
        # from https://github.com/arpitbansal297/Universal-Guided-Diffusion/blob/main/stable-diffusion-guided/scripts/clip_guided.py#L172
        self.trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # mean = [0.48145466, 0.4578275, 0.40821073], std = [0.26862954, 0.26130258, 0.27577711]
    
    def load_concept_embeddings(self, concepts):
        """
        Retrieves CLIP text embeddings for selected concepts
        
        Arguments:
            concepts: List[str]
                List of strings with phrases/words to detect and unguide diffusion generation
        """
        text_input = clip.tokenize(concepts).to(self.device)
        self.concept_embeds = self.clip_model.encode_text(text_input)
    
    def load_base_embedding(self, base_concepts):
        """
        Retrieves CLIP text embeddings for base concepts.
        These will be used to compare against concept embeddings to determine the precense of the latter.
        
        Arguments:
            base_concepts: List[str]
                List of strings with phrases/words to detect and unguide diffusion generation
        
        To do: generalise to image inputs.
        """
        text_input = clip.tokenize(base_concepts).to(self.device)
        self.base_embeds = self.clip_model.encode_text(text_input)
    
    @staticmethod
    def cosine_similarity(image_embeds:torch.Tensor, text_embeds:torch.Tensor):
        """Computes the cosine similarity between (CLIP) embeddings"""
        normalized_image_embeds = nn.functional.normalize(image_embeds)
        normalized_text_embeds = nn.functional.normalize(text_embeds)
        return torch.mm(normalized_image_embeds, normalized_text_embeds.t())
    
    def preprocess(self, x_input:torch.Tensor):
        """Preprocess a tensor image
        Arguments
            x_input: torch.Tensor
                input tensor of size (b, c=3, h, w) with a batch of images to process
            
        Returns
            x_out = torch.Tensor of size (b, c=3, 224, 224)
        """
        x_out = (x_input + 1) * 0.5
        x_out = F.resize(x_out, (224, 224), interpolation=F.InterpolationMode.BICUBIC)
        x_out = self.trans(x_out)
        return x_out
    
    def preprocess_numpy(self,x_input):
        """Equivalent to preprocess but for a batch of type numpy.array (shape (b,c,h,w))"""
        cliped_imgs = []
        for x_img in x_input:
            x_img = 255. * rearrange(x_img, 'c h w -> h w c')
            pil_img = Image.fromarray(x_img.astype(np.uint8))
            cliped_imgs.append(self.preprocess_clip(pil_img))
        return torch.stack(cliped_imgs,dim=0)

    def forward(self, clip_input):
        """
        Returns a set of cosine similarities given of the input, with respect to a set of CLIP embeddings.

        Arguments:
            clip_input: transformers.feature_extraction_utils.BatchFeature
                RGB image in CLIP format
            
            verbose: bool, default = False
                Whether to print details (similarity, threshold and results) for each sensitive concept

        Returns:
            probs: numpy.array
                probsbility for the target concept
        """
        clip_input = clip_input.to(self.device)
        image_embeds = self.clip_model.encode_image(clip_input)
        embeds = torch.concat([self.concept_embeds,self.base_embeds])
        probs = self.cosine_similarity(image_embeds, embeds)
        probs = probs.softmax(dim=-1)
        probs = probs[:,0].unsqueeze(1)
        return probs
