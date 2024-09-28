import torch
from extractors import *
from encoders import *

def get_class(class_name):
    return globals()[class_name]


class Pipeline:
    def __init__(self, extractor, encoder, pool=False):
        self.extractor = extractor
        self.encoder = encoder
        self.pool = pool

    def __call__(self, file_path):
        embeddings = []
        fragments = self.extractor(file_path)
        for fragment in fragments:
            fragment_embedding = self.encoder.encode(**fragment).squeeze().detach().cpu()
            if fragment_embedding.dim() == 1:
                embeddings.append(fragment_embedding)
            else:
                embeddings.extend(fragment_embedding)

        embeddings = torch.stack(embeddings)
        if self.pool:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)
        
        return embeddings.numpy()
    
    @staticmethod
    def from_yaml(yaml_path):
        import yaml
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        
        extractor = get_class(config["extractor"]["name"])(**config["extractor"]["args"])
        encoder = get_class(config["encoder"]["name"])(**config["encoder"]["args"])
        pool = config["pool"]
        return Pipeline(extractor, encoder, pool)

