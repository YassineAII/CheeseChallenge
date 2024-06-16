from .base import DatasetGenerator
from generators.dreambooth import DreamBoothGenerator
from tqdm import tqdm
import random

def read_label_list(file_path):
    with open(file_path, 'r') as file:
        labels = [line.strip() for line in file]
    return labels

random_style = ["a photo"]*120 + [
    "a sketch", "a painting", "an illustration", "a digital rendering", "a print", "a comic-style depiction",
    "a manga-style depiction", "a pixel art representation", "a tattoo design", "a graffiti-style portrayal",
    "a watercolor", "an oil painting", "a charcoal drawing", "a pastel drawing", "a stencil art", "a collage",
    "a mosaic", "a silhouette", "a pop art version", "a sculpture", "an origami", "an embroidery", "a quilt pattern",
    "a stained glass design", "a woodcut", "an etching", "a lithograph", "a screen print", "a relief carving",
    "a bronze casting", "a glass blowing", "a ceramic pottery", "a tapestry", "a fresco", "a mural", "a doodle",
    "a cartoon", "an animation", "a 3D model", "a wireframe model", "a CGI", "a virtual reality model",
    "an augmented reality model", "a hologram", "a gouache painting", "an ink wash painting", "a digital painting",
    "a stencil graffiti", "an airbrush art", "a pointillism", "an impasto painting", "an engraving", "a linocut",
    "a marquetry", "a papercut", "a batik design", "a cross-stitch pattern", "a macramé design", "a beadwork design",
    "a sand sculpture"
]


class GPTDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        labels_path,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=2000,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label
        self.labels = read_label_list(labels_path)

    def create_prompts(self, labels_names):
        prompts =  {"BRIE DE MELUN": [
    {
      "prompt": "A photo of Brie de Melun, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Brie de Melun, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Brie de Melun, in a sandwich with ham, picnic blanket, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Brie de Melun, melted on a slice of bread, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Brie de Melun, with wine glass and nuts, wooden table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Brie de Melun, on a cracker with honey, dining table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Brie de Melun, with strawberries and bread, breakfast table, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Brie de Melun, in a cheese fondue pot, dining table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Brie de Melun, with figs and walnuts, cheese platter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Brie de Melun, served with a glass of champagne, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "CAMEMBERT": [
    {
      "prompt": "A photo of Camembert, baked with herbs, ceramic dish, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Camembert, on a cheese board with crackers, wooden table, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Camembert, in a sandwich with lettuce, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Camembert, melted on toast, white plate, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Camembert, served with apple slices, rustic wooden table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Camembert, wrapped in puff pastry, baking tray, oven light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Camembert, on a charcuterie board with meats, picnic blanket, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Camembert, with honey drizzled on top, white plate, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Camembert, in a salad with mixed greens, wooden bowl, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Camembert, served with a glass of white wine, elegant table setting, natural sunlight, close-up shot",
      "num_images": 10
    }
  ],
  "EPOISSES": [
    {
      "prompt": "A photo of Epoisses, in its wooden box, rustic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Epoisses, with sliced apples and nuts, cheese platter, bright indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Epoisses, on a cracker with honey, dining table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Epoisses, spread on a baguette, kitchen counter, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Epoisses, with a glass of white wine, rustic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Epoisses, in a salad with walnuts and grapes, ceramic bowl, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Epoisses, with crackers and figs, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Epoisses, on a cheese platter with strawberries, picnic blanket, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Epoisses, melted over potatoes, rustic table, candlelight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Epoisses, served with a glass of red wine, elegant table setting, natural daylight, close-up shot",
      "num_images": 10
    }
  ],
  "FOURME D’AMBERT": [
    {
      "prompt": "A photo of Fourme d’Ambert, crumbled on a salad, ceramic bowl, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fourme d’Ambert, with figs and walnuts, cheese platter, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fourme d’Ambert, in a sandwich with arugula, picnic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fourme d’Ambert, melted on a steak, white plate, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fourme d’Ambert, with a glass of red wine, rustic wooden table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fourme d’Ambert, in a quiche, baking dish, oven light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fourme d’Ambert, with crackers and pear slices, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fourme d’Ambert, on a cheese platter with grapes, picnic blanket, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fourme d’Ambert, melted over vegetables, rustic table, candlelight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fourme d’Ambert, served with a glass of port wine, elegant table setting, natural daylight, close-up shot",
      "num_images": 10
    }
  ],
  "RACLETTE": [
    {
      "prompt": "A photo of Raclette, melted over potatoes, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Raclette, with pickles and onions, serving platter, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Raclette, in a sandwich with ham, picnic blanket, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Raclette, melted on a baguette, kitchen counter, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Raclette, with a glass of white wine, rustic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Raclette, on a cheese board with meats, wooden table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Raclette, with roasted vegetables, ceramic plate, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Raclette, melted on a grilled sandwich, picnic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Raclette, in a cheese fondue pot, rustic table, candlelight, medium shot",
      "num_images": 10


    },
    {
      "prompt": "A photo of Raclette, served with a glass of rosé wine, elegant table setting, natural daylight, close-up shot",
      "num_images": 10
    }
  ],
  "MORBIER": [
    {
      "prompt": "A photo of Morbier, on a cheese board with fruits, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Morbier, melted on bread, kitchen counter, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Morbier, with nuts and figs, cheese platter, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Morbier, in a sandwich with lettuce, picnic table, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Morbier, served with honey and almonds, rustic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Morbier, melted in a panini, ceramic plate, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Morbier, on a cheese platter with crackers, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Morbier, with apple slices and walnuts, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Morbier, in a grilled cheese sandwich, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Morbier, served with a glass of cider, rustic table, natural daylight, close-up shot",
      "num_images": 10
    }
  ],
  "SAINT-NECTAIRE": [
    {
      "prompt": "A photo of Saint-Nectaire, on a cheese board with nuts, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Nectaire, melted on a baguette, kitchen counter, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Nectaire, with figs and walnuts, cheese platter, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Nectaire, in a sandwich with arugula, picnic table, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Nectaire, with a glass of white wine, rustic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Nectaire, on a cheese board with grapes, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Nectaire, melted on roasted vegetables, ceramic plate, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Nectaire, in a grilled cheese sandwich, picnic blanket, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Nectaire, with pear slices and walnuts, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Nectaire, served with a glass of Chardonnay, elegant table setting, natural daylight, close-up shot",
      "num_images": 10
    }
  ],
  "POULIGNY SAINT-PIERRE": [
    {
      "prompt": "A photo of Pouligny Saint-Pierre, on a cheese board with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pouligny Saint-Pierre, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pouligny Saint-Pierre, in a salad with walnuts, ceramic bowl, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pouligny Saint-Pierre, with honey and nuts, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pouligny Saint-Pierre, with a glass of white wine, rustic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pouligny Saint-Pierre, on a cheese platter with strawberries, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pouligny Saint-Pierre, with figs and crackers, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pouligny Saint-Pierre, in a sandwich with arugula, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pouligny Saint-Pierre, melted over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pouligny Saint-Pierre, served with a glass of Sauvignon Blanc, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "ROQUEFORT": [
    {
      "prompt": "A photo of Roquefort, crumbled on a salad, ceramic bowl, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Roquefort, with figs and walnuts, cheese platter, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Roquefort, in a sandwich with arugula, picnic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Roquefort, melted on a steak, white plate, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Roquefort, with a glass of red wine, rustic wooden table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Roquefort, in a quiche, baking dish, oven light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Roquefort, with crackers and pear slices, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Roquefort, on a cheese platter with grapes, picnic blanket, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Roquefort, melted over vegetables, rustic table, candlelight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Roquefort, served with a glass of port wine, elegant table setting, natural daylight, close-up shot",
      "num_images": 10
    }
  ],
  "COMTÉ": [
    {
      "prompt": "A photo of Comté, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Comté, in a sandwich with ham, picnic blanket, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Comté, melted on a slice of bread, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Comté, with wine glass and nuts, wooden table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Comté, on a cracker with honey, dining table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Comté, with strawberries and bread, breakfast table, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Comté, in a cheese fondue pot, dining table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Comté, with figs and walnuts, cheese platter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Comté, served with a glass of champagne, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Comté, melted over potatoes, rustic table, natural sunlight, close-up shot",
      "num_images": 10
    }
  ],
  "CHÈVRE": [
    {
      "prompt": "A photo of Chèvre, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chèvre, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chèvre, in a salad with walnuts, ceramic bowl, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chèvre, with honey and nuts, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chèvre, with a glass of white wine, rustic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chèvre, on a cheese platter with strawberries, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chèvre, with figs and crackers, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chèvre, in a sandwich with arugula, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chèvre, melted over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chèvre, served with a glass of Sauvignon Blanc, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "PECORINO": [
    {
      "prompt": "A photo of Pecorino, grated over pasta, white plate, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pecorino, on a cheese board with nuts, wooden table, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pecorino, with figs and honey, cheese platter, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pecorino, in a salad with arugula, ceramic bowl, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pecorino, served with a glass of red wine, rustic wooden table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pecorino, on a charcuterie board with meats, wooden table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pecorino, with crackers and olives, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pecorino, in a sandwich with tomatoes, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pecorino, grated over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Pecorino, served with a glass of Chianti, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "NEUFCHATEL": [
    {
      "prompt": "A photo of Neufchatel, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Neufchatel, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Neufchatel, in a sandwich with ham, picnic blanket, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Neufchatel, melted on a slice of bread, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Neufchatel, with wine glass and nuts, wooden table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Neufchatel, on a cracker with honey, dining table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Neufchatel, with strawberries and bread, breakfast table, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Neufchatel, in a cheese fondue pot, dining table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Neufchatel, with figs and walnuts, cheese platter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Neufchatel, served with a glass of champagne, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "CHEDDAR": [
    {
      "prompt": "A photo of Cheddar, sliced on a wooden cutting board, kitchen counter, bright indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cheddar, block with a cheese knife, rustic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cheddar, in a grilled cheese sandwich, white plate, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cheddar, cubed with toothpicks, party platter, bright indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cheddar, melted on a burger, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cheddar, shredded on a pizza, pizza tray, oven light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cheddar, in a cheese fondue pot, dining table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cheddar, with grapes and crackers, cheese board, natural sunlight, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cheddar, stacked slices, sandwich prep station, kitchen light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cheddar, with wine glass, picnic blanket, natural daylight, close-up shot",
      "num_images": 10
    }
  ],
  "BÛCHETTE DE CHÈVRE": [
    {
      "prompt": "A photo of Bûchette de Chèvre, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Bûchette de Chèvre, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Bûchette de Chèvre, in a salad with walnuts, ceramic bowl, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Bûchette de Chèvre, with honey and nuts, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Bûchette de Chèvre, with a glass of white wine, rustic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Bûchette de Chèvre, on a cheese platter with strawberries, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Bûchette de Chèvre, with figs and crackers, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Bûchette de Chèvre, in a sandwich with arugula, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Bûchette de Chèvre, melted over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Bûchette de Chèvre, served with a glass of Sauvignon Blanc, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "PARMESAN": [
    {
      "prompt": "A photo of Parmesan, grated over pasta, white plate, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Parmesan, on a cheese board with nuts, wooden table, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Parmesan, with figs and honey, cheese platter, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Parmesan, in a salad with arugula, ceramic bowl, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Parmesan, served with a glass of red wine, rustic wooden table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Parmesan, on a charcuterie board with meats, wooden table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Parmesan, with crackers and olives, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Parmesan, in a sandwich with tomatoes, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Parmesan, grated over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Parmesan, served with a glass of Chianti, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "SAINT-FÉLICIEN": [
    {
      "prompt": "A photo of Saint-Félicien, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Félicien, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Félicien, in a salad with walnuts, ceramic bowl, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Félicien, with honey and nuts, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Félicien, with a glass of white wine, rustic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Félicien, on a cheese platter with strawberries, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Félicien, with figs and crackers, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Félicien, in a sandwich with arugula, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Félicien, melted over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Saint-Félicien, served with a glass of Sauvignon Blanc, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "MONT D’OR": [
    {
      "prompt": "A photo of Mont d’Or, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mont d’Or, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mont d’Or, in a salad with walnuts, ceramic bowl, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mont d’Or, with honey and nuts, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mont d’Or, with a glass of white wine, rustic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mont d’Or, on a cheese platter with strawberries, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mont d’Or, with figs and crackers, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mont d’Or, in a sandwich with arugula, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mont d’Or, melted over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mont d’Or, served with a glass of Sauvignon Blanc, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "STILTON": [
    {
      "prompt": "A photo of Stilton, crumbled on a salad, ceramic bowl, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Stilton, with figs and walnuts, cheese platter, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Stilton, in a sandwich with arugula, picnic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Stilton, melted on a steak, white plate, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Stilton, with a glass of red wine, rustic wooden table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Stilton, in a quiche, baking dish, oven light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Stilton, with crackers and pear slices, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Stilton, on a cheese platter with grapes, picnic blanket, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Stilton, melted over vegetables, rustic table, candlelight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Stilton, served with a glass of port wine, elegant table setting, natural daylight, close-up shot",
      "num_images": 10
    }
  ],
  "SCARMOZA": [
    {
      "prompt": "A photo of Scarmoza, melted on a pizza, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Scarmoza, sliced on a sandwich, picnic blanket, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Scarmoza, on a cheese platter with grapes, rustic wooden table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Scarmoza, with olives and breadsticks, cheese board, natural sunlight, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Scarmoza, melted over roasted vegetables, rustic table, candlelight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Scarmoza, in a grilled cheese sandwich, ceramic plate, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Scarmoza, on a charcuterie board with meats, wooden table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Scarmoza, with figs and honey, cheese platter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Scarmoza, in a sandwich with arugula, picnic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Scarmoza, served with a glass of red wine, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "CABECOU": [
    {
      "prompt": "A photo of Cabecou, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cabecou, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cabecou, in a salad with walnuts, ceramic bowl, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cabecou, with honey and nuts, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cabecou, with a glass of white wine, rustic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cabecou, on a cheese platter with strawberries, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cabecou, with figs and crackers, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cabecou, in a sandwich with arugula, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cabecou, melted over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Cabecou, served with a glass of Sauvignon Blanc, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "BEAUFORT": [
    {
      "prompt": "A photo of Beaufort, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Beaufort, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Beaufort, in a salad with walnuts, ceramic bowl, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Beaufort, with honey and nuts, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Beaufort, with a glass of white wine, rustic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Beaufort, on a cheese platter with strawberries, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Beaufort, with figs and crackers, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Beaufort, in a sandwich with arugula, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Beaufort, melted over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Beaufort, served with a glass of Sauvignon Blanc, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "MUNSTER": [
    {
      "prompt": "A photo of Munster, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Munster, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Munster, in a salad with walnuts, ceramic bowl, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Munster, with honey and nuts, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Munster, with a glass of white wine, rustic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Munster, on a cheese platter with strawberries, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Munster, with figs and crackers, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Munster, in a sandwich with arugula, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Munster, melted over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Munster, served with a glass of Sauvignon Blanc, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "CHABICHOU": [
    {
      "prompt": "A photo of Chabichou, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chabichou, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chabichou, in a salad with walnuts, ceramic bowl, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chabichou, with honey and nuts, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chabichou, with a glass of white wine, rustic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chabichou, on a cheese platter with strawberries, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chabichou, with figs and crackers, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chabichou, in a sandwich with arugula, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chabichou, melted over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Chabichou, served with a glass of Sauvignon Blanc, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "TOMME DE VACHE": [
    {
      "prompt": "A photo of Tomme de Vache, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tomme de Vache, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tomme de Vache, in a salad with walnuts, ceramic bowl, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tomme de Vache, with honey and nuts, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tomme de Vache, with a glass of white wine, rustic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tomme de Vache, on a cheese platter with strawberries, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tomme de Vache, with figs and crackers, cheese board, indoor light, top-down angle",
      "num_images": 10


    },
    {
      "prompt": "A photo of Tomme de Vache, in a sandwich with arugula, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tomme de Vache, melted over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tomme de Vache, served with a glass of Sauvignon Blanc, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "REBLOCHON": [
    {
      "prompt": "A photo of Reblochon, on a cheese platter with grapes, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Reblochon, spread on a baguette slice, kitchen counter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Reblochon, in a salad with walnuts, ceramic bowl, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Reblochon, with honey and nuts, dining table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Reblochon, with a glass of white wine, rustic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Reblochon, on a cheese platter with strawberries, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Reblochon, with figs and crackers, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Reblochon, in a sandwich with arugula, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Reblochon, melted over roasted vegetables, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Reblochon, served with a glass of Sauvignon Blanc, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "EMMENTAL": [
    {
      "prompt": "A photo of Emmental, sliced on a wooden cutting board, kitchen counter, bright indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Emmental, block with a cheese knife, rustic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Emmental, in a grilled cheese sandwich, white plate, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Emmental, cubed with toothpicks, party platter, bright indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Emmental, melted on a burger, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Emmental, shredded on a pizza, pizza tray, oven light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Emmental, in a cheese fondue pot, dining table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Emmental, with grapes and crackers, cheese board, natural sunlight, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Emmental, stacked slices, sandwich prep station, kitchen light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Emmental, with wine glass, picnic blanket, natural daylight, close-up shot",
      "num_images": 10
    }
  ],
  "FETA": [
    {
      "prompt": "A photo of Feta, crumbled on a Greek salad, ceramic bowl, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Feta, with olives and tomatoes, cheese platter, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Feta, in a sandwich with cucumbers, picnic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Feta, drizzled with olive oil, white plate, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Feta, with a glass of white wine, rustic wooden table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Feta, in a baked dish with tomatoes, ceramic baking dish, oven light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Feta, on a cheese board with herbs, wooden table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Feta, with crackers and grapes, cheese board, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Feta, in a wrap with vegetables, picnic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Feta, served with a glass of Chardonnay, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "OSSAU-IRATY": [
    {
      "prompt": "A photo of Ossau-Iraty, on a cheese board with figs, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Ossau-Iraty, sliced on a sandwich, picnic blanket, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Ossau-Iraty, with olives and nuts, cheese platter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Ossau-Iraty, melted over roasted potatoes, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Ossau-Iraty, served with a glass of red wine, wooden table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Ossau-Iraty, on a cheese platter with grapes, kitchen counter, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Ossau-Iraty, with honey drizzled on top, white plate, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Ossau-Iraty, in a salad with mixed greens, wooden bowl, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Ossau-Iraty, with sliced bread and butter, rustic table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Ossau-Iraty, served with a glass of Pinot Noir, elegant table setting, natural sunlight, medium shot",
      "num_images": 10
    }
  ],
  "MIMOLETTE": [
    {
      "prompt": "A photo of Mimolette, sliced on a cheese platter, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mimolette, grated over pasta, white plate, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mimolette, with figs and walnuts, cheese board, natural sunlight, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mimolette, melted in a sandwich, picnic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mimolette, with a glass of red wine, kitchen counter, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mimolette, on a charcuterie board with meats, wooden table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mimolette, with crackers and grapes, cheese platter, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mimolette, in a salad with mixed greens, ceramic bowl, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mimolette, with honey drizzled on top, rustic table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mimolette, served with a glass of Chardonnay, elegant table setting, natural sunlight, medium shot",
      "num_images": 10
    }
  ],
  "MAROILLES": [
    {
      "prompt": "A photo of Maroilles, on a cheese board with figs, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Maroilles, sliced on a sandwich, picnic blanket, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Maroilles, with olives and nuts, cheese platter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Maroilles, melted over roasted potatoes, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Maroilles, served with a glass of red wine, wooden table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Maroilles, on a cheese platter with grapes, kitchen counter, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Maroilles, with honey drizzled on top, white plate, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Maroilles, in a salad with mixed greens, wooden bowl, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Maroilles, with sliced bread and butter, rustic table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Maroilles, served with a glass of Pinot Noir, elegant table setting, natural sunlight, medium shot",
      "num_images": 10
    }
  ],
  "GRUYÈRE": [
    {
      "prompt": "A photo of Gruyère, sliced on a wooden cutting board, kitchen counter, bright indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Gruyère, block with a cheese knife, rustic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Gruyère, in a grilled cheese sandwich, white plate, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Gruyère, cubed with toothpicks, party platter, bright indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Gruyère, melted on a burger, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Gruyère, shredded on a pizza, pizza tray, oven light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Gruyère, in a cheese fondue pot, dining table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Gruyère, with grapes and crackers, cheese board, natural sunlight, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Gruyère, stacked slices, sandwich prep station, kitchen light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Gruyère, with wine glass, picnic blanket, natural daylight, close-up shot",
      "num_images": 10
    }
  ],
  "MOTHAIS": [
    {
      "prompt": "A photo of Mothais, on a cheese board with figs, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mothais, sliced on a sandwich, picnic blanket, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mothais, with olives and nuts, cheese platter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mothais, melted over roasted potatoes, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mothais, served with a glass of red wine, wooden table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mothais, on a cheese platter with grapes, kitchen counter, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mothais, with honey drizzled on top, white plate, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mothais, in a salad with mixed greens, wooden bowl, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mothais, with sliced bread and butter, rustic table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mothais, served with a glass of Pinot Noir, elegant table setting, natural sunlight, medium shot",
      "num_images": 10
    }
  ],
  "VACHERIN": [
    {
      "prompt": "A photo of Vacherin, on a cheese board with figs, rustic wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Vacherin, sliced on a sandwich, picnic blanket, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Vacherin, with olives and nuts, cheese platter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Vacherin, melted over roasted potatoes, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Vacherin, served with a glass of red wine, wooden table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Vacherin, on a cheese platter with grapes, kitchen counter, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Vacherin, with honey drizzled on top, white plate, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Vacherin, in a salad with mixed greens, wooden bowl, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Vacherin, with sliced bread and butter, rustic table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Vacherin, served with a glass of Pinot Noir, elegant table setting, natural sunlight, medium shot",
      "num_images": 10
    }
  ],
  "MOZZARELLA": [
    {
      "prompt": "A photo of Mozzarella, sliced on a Caprese salad, white plate, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mozzarella, melted on a pizza, wooden table, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mozzarella, with tomatoes and basil, cheese board, natural sunlight, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mozzarella, in a sandwich with arugula, picnic table, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mozzarella, with a glass of white wine, kitchen counter, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mozzarella, in a baked pasta dish, ceramic baking dish, oven light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mozzarella, on a cheese board with olives, wooden table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mozzarella, with crackers and grapes, cheese platter, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mozzarella, in a wrap with vegetables, picnic table, natural sunlight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Mozzarella, served with a glass of Chardonnay, elegant table setting, natural daylight, close-up shot",
      "num_images": 10
    }
  ],
  "TÊTE DE MOINES": [
    {
      "prompt": "A photo of Tête de Moines, shaved into rosettes, wooden table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tête de Moines, on a cheese board with grapes, rustic wooden table, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tête de Moines, with figs and nuts, cheese platter, natural sunlight, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tête de Moines, in a salad with arugula, ceramic bowl, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tête de Moines, with a glass of red wine, kitchen counter, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tête de Moines, on a charcuterie board with meats, wooden table, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tête de Moines, with honey drizzled on top, white plate, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tête de Moines, in a sandwich with tomatoes, picnic table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tête de Moines, served with sliced bread and butter, rustic table, candlelight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Tête de Moines, served with a glass of Pinot Noir, elegant table setting, natural daylight, medium shot",
      "num_images": 10
    }
  ],
  "FROMAGE FRAIS": [
    {
      "prompt": "A photo of Fromage Frais, in a bowl with fresh berries, kitchen table, natural daylight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fromage Frais, spread on a baguette slice, rustic wooden table, indoor light, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fromage Frais, in a salad with mixed greens, ceramic bowl, natural sunlight, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fromage Frais, with honey and walnuts, breakfast table, indoor light, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fromage Frais, served with a glass of orange juice, kitchen counter, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fromage Frais, in a parfait with granola, clear glass, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fromage Frais, on a cheese board with crackers, wooden table, natural sunlight, top-down angle",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fromage Frais, with sliced fruit and nuts, breakfast table, indoor light, close-up shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fromage Frais, in a wrap with vegetables, picnic table, natural daylight, medium shot",
      "num_images": 10
    },
    {
      "prompt": "A photo of Fromage Frais, served with a cup of tea, elegant table setting, natural sunlight, close-up shot",
      "num_images": 10
    }
  ]
}

        selected_prompts = {k: prompts[k] for k in self.labels}

        return selected_prompts


    def generate(self, labels_names):
        labels_prompts = self.create_prompts(labels_names)
        for label, label_prompts in labels_prompts.items():
            self.generator = DreamBoothGenerator(label = label)
            image_id_0 = 0
            for prompt_metadata in label_prompts:
                num_images_per_prompt = prompt_metadata["num_images"]
                p = prompt_metadata["prompt"].lower().replace("a photo", random_style.choice())
                prompt = [p] * num_images_per_prompt
                pbar = tqdm(range(0, num_images_per_prompt, self.batch_size))
                pbar.set_description(
                    f"Generating images for prompt: {prompt_metadata['prompt']}"
                )
                for i in range(0, num_images_per_prompt, self.batch_size):
                    batch = prompt[i : i + self.batch_size]
                    images = self.generator.generate(batch)
                    self.save_images(images, label, image_id_0)
                    image_id_0 += len(images)
                    pbar.update(1)
                pbar.close()
