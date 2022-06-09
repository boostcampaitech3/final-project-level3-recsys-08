import torch

recipe_cos = torch.load('/opt/ml/recipe/product_serving/final-project-level3-recsys-08/HGAT/sentence_emb/sentence_5900_cos_multilingual-cased-v1.pt')

def get_similar_recipe(r_idx, recipe_cos, topk):
        
    return recipe_cos[r_idx].topk(topk).indices
    