from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from fuzzywuzzy import fuzz, process
import unicodedata
import os, json




ocr = ocr_predictor(pretrained=True)


def get_names(file):
    with open(file, "r") as f:
        names = f.readlines()
        names = [name.strip().upper() for name in names]
    return names


cheese_names = get_names("ist_of_cheese.txt")
        
def format_text(text):
    nfkd_form = unicodedata.normalize('NFKD', text)
    formated_text = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

    return formated_text.upper()

possibilities =  {
        "TOMME DE VACHE": ["TOMME"],
        "BRIE DE MELUN": ['BRIE', 'MELUN'],
        "MOZZARELLA": ["MOZZA", "MOZARELISSIMA"],
        "OSSAU- IRATY": ["OSSAU-IRATY", "OSSAU", "IRATY"],
        "COMTÉ": ["COMTE"],
        "FOURME D’AMBERT": ["FOURME", "AMBERT"],
        "FROMAGE FRAIS": ["FRAIS"],
        "SAINT-NECTAIRE": ["SAINT-NECTAIRE","NECTAIRE"],
        "GRUYÈRE": ["GRUYERE"],
        "MONT D’OR": ["MONT", "OR", "D'OR"],
        "POULIGNY SAINT- PIERRE": ["POULIGNY", "PIERRE", "SAINT-PIERRE"],
        "SAINT- FÉLICIEN": ["SAINT-FELICIEN","FELICIEN"],
        "TÊTE DE MOINES": ["TETE", "MOINES", "MOINE"],
        "FETA": ["GREC", "GREEK", "GREECE"],
        
    }



def find_possibilities(word):
    for key, values in possibilities.items():
        if word in values:
            return key, True
    return word, False


def read_image(image_path):
    image = DocumentFile.from_images(image_path)
    result = ocr(image)
    return result


def clean_text(result):
    lines = format_text(result.render())
    lines = lines.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    words = [word for line in lines for word in line.split()]

    return words

def from_words_to_cheese(words):
    dict_cheese_name_rate = {}
    for word in words:
        possible_word, is_possibility = find_possibilities(word)
        if is_possibility:
            dict_cheese_name_rate[possible_word] = 1.0
        else:
            match, rate = process.extractOne(possible_word, cheese_names, scorer=fuzz.ratio)
            if rate > 83:  
                dict_cheese_name_rate[match] = rate / 100

    return dict_cheese_name_rate

def from_image_to_cheese(image_path):
    words = clean_text(read_image(image_path))
    dict_image_to_possible_cheese_names = from_words_to_cheese(words)
    return dict_image_to_possible_cheese_names


detected_cheese = {}
dataset_path = "/users/eleves-b/2022/elias.ben-rhouma/Desktop/cheese_classification_challenge-master/dataset/test"
dataset = os.listdir(dataset_path)
for i, image in enumerate(dataset):
    if i % 20 == 0:
        print(f"{i}/{len(dataset)}")
    image_path = os.path.join(dataset_path, image)
    detected_cheese[image] = from_image_to_cheese(image_path)



with open("ocrtest.json", "w") as file:
    json.dump(detected_cheese, file)


