import hydra
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torch
import difflib
from unidecode import unidecode
from typing import List, Tuple
import numpy as np
import pytesseract
import cv2
import json
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "/users/eleves-a/2022/yassine.laraki/Desktop/INF473VV/Cheese-Challenge/ocr_preds.pkl"

with open(file_path, 'rb') as f:
    ocr_predictions = pickle.load(f)


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)

def calculate_substring_similarity(text: str, cheese_names: List[str]) -> List[Tuple[str, float]]:
    results = []
    text = unidecode(text).lower()
    text_len = len(text)
    best_cheese = None
    best_sim = 0.0
    best_cheese = None
    for ch in cheese_names:
        cheese = unidecode(ch)
        cheese_lower = cheese.lower()
        cheese_len = len(cheese_lower)
        
        
        # Glisser une fenêtre de longueur égale au nom du fromage sur le texte OCR
        for i in range(text_len - cheese_len + 1):
            substring = text[i:i + cheese_len]
            similarity = difflib.SequenceMatcher(None, substring, cheese_lower).ratio()
            if similarity > best_sim:
                best_sim = similarity
                best_cheese = ch

    return(best_cheese,best_sim)

class_to_idx = {'BEAUFORT': 0, 'BRIE DE MELUN': 1, 'BÛCHETTE DE CHÈVRE': 2, 'CABECOU': 3, 'CAMEMBERT': 4, 'CHABICHOU': 5, 'CHEDDAR': 6, 'CHÈVRE': 7, 'COMTÉ': 8, 'EMMENTAL': 9, 'EPOISSES': 10, 'FETA': 11, 'FOURME D’AMBERT': 12, 'FROMAGE FRAIS': 13, 'GRUYÈRE': 14, 'MAROILLES': 15, 'MIMOLETTE': 16, 'MONT D’OR': 17, 'MORBIER': 18, 'MOTHAIS': 19, 'MOZZARELLA': 20, 'MUNSTER': 21, 'NEUFCHATEL': 22, 'OSSAU- IRATY': 23, 'PARMESAN': 24, 'PECORINO': 25, 'POULIGNY SAINT- PIERRE': 26, 'RACLETTE': 27, 'REBLOCHON': 28, 'ROQUEFORT': 29, 'SAINT- FÉLICIEN': 30, 'SAINT-NECTAIRE': 31, 'SCARMOZA': 32, 'STILTON': 33, 'TOMME DE VACHE': 34, 'TÊTE DE MOINES': 35, 'VACHERIN': 36}

@hydra.main(config_path="configs/train", config_name="config")
def create_submission(cfg):
    test_loader = DataLoader(
        TestDataset(
            cfg.dataset.test_path, hydra.utils.instantiate(cfg.dataset.test_transform)
        ),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )
    # Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    class_names = sorted(os.listdir(cfg.dataset.train_path))
    if ".DS_Store" in class_names:
        class_names.remove(".DS_Store")

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])
    nb_ocr = 0
    for i, batch in enumerate(test_loader):
        images, image_names = batch
        images = images.to(device)
        preds = model(images)
        ocr_fromages = ocr_predictions[image_names[0]+".jpg"]
        
        if ocr_fromages == {}:
            preds = preds.argmax(1)
            preds = [class_names[pred] for pred in preds.cpu().numpy()]
        else:
            nb_ocr += 1
            best_fromage = None
            best_prob = 0.0
            for fromage in ocr_fromages.keys():
                p = class_to_idx[fromage]
                if p >= best_prob:
                    best_prob = p
                    best_fromage = fromage
            
            preds = [best_fromage]
        
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "label": preds}),
            ]
        )
        
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)

if __name__ == "__main__":
    create_submission()