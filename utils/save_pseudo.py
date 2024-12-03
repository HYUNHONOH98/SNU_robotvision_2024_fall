from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np

palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
palette_label = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
 "traffic sign", "vegetation", "terrain","sky", "person", "rider", "car",
   "truck", "bus", "train","motorcycle", "bicycle"]

cityscape_image_mean = np.array([0.4422, 0.4379, 0.4246])
cityscape_image_std = np.array([0.2572, 0.2516, 0.2467])

def save_pseudo_labels(names, originals, pseudo_labels, save_dir, threshold):
  for idx, name in enumerate(names):
    img_name = name.split("/")[-1]
    if img_name in [
      "aachen_000000_000019_leftImg8bit.png",
      "bochum_000000_000313_leftImg8bit.png",
      "bremen_000000_000019_leftImg8bit.png",
      "cologne_000000_000019_leftImg8bit.png",
      "darmstadt_000000_000019_leftImg8bit.png",
      "dusseldorf_000000_000019_leftImg8bit.png",
      "erfurt_000000_000019_leftImg8bit.png",
      "hamburg_000000_000019_leftImg8bit.png",
      "hanover_000000_000019_leftImg8bit.png",
      "jena_000000_000019_leftImg8bit.png",
      "krefeld_000000_000019_leftImg8bit.png",
      "monchengladbach_000000_000019_leftImg8bit.png",
      "strasbourg_000000_000019_leftImg8bit.png",
      "tubingen_000000_000019_leftImg8bit.png",
      "ulm_000000_000019_leftImg8bit.png",
      "weimar_000000_000019_leftImg8bit.png",
      "zurich_000000_000019_leftImg8bit.png",
    ]:
      save_path = os.path.join(save_dir, img_name)
      original_image = originals[idx].permute(1, 2, 0).cpu().numpy() 
      original_image = (original_image * cityscape_image_std + cityscape_image_mean) * 255
      original_image = np.clip(original_image, 0, 255).astype(np.uint8)

      title = ""

      threshold = threshold.tolist()
      for i in range(len(threshold)):
        t = round(threshold[i],2)
        p_l = palette_label[i]

        if i % 4 == 3:
          title += f"{p_l} : {t}\n"
        else:
          title += f"{p_l} : {t}, "

      tensor = pseudo_labels[idx].byte()
      array = tensor.cpu().numpy()
      height, width = array.shape
      rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

      for class_id, color in enumerate(palette):
        rgb_image[array == class_id] = color
      rgb_image[array == 255] = [255, 255, 255]

        # Create figure with subplots

      fig, axes = plt.subplots(1, 2, figsize=(12, 6))
      fig.suptitle(title, fontsize=12)
      
      # Original image subplot
      axes[0].imshow(original_image)
      axes[0].axis('off')
      axes[0].set_title("Original Image")
      
      # Pseudo label subplot
      axes[1].imshow(rgb_image)
      axes[1].axis('off')
      axes[1].set_title("Pseudo Label")
      
      # Save the combined figure
      plt.tight_layout()
      plt.subplots_adjust(top=0.85)  # Adjust top margin for title
      plt.savefig(save_path, dpi=300)
      plt.close()

  return


def save_valid_label(names, pseudo_labels, save_dir):
  for idx, name in enumerate(names):
    img_name = name.split("/")[-1]
    if img_name in ["frankfurt_000001_052594_leftImg8bit.png",
     "frankfurt_000001_049209_leftImg8bit.png",
     "frankfurt_000000_001751_leftImg8bit.png",
     "frankfurt_000001_012519_leftImg8bit.png",
     "munster_000084_000019_leftImg8bit.png",
     "munster_000003_000019_leftImg8bit.png",
     "munster_000037_000019_leftImg8bit.png",
     "lindau_000056_000019_leftImg8bit.png",
     "lindau_000022_000019_leftImg8bit.png",
     "lindau_000025_000019_leftImg8bit.png"
     ]:
      save_path = os.path.join(save_dir, img_name)

      tensor = pseudo_labels[idx].byte()
      array = tensor.cpu().numpy()
      height, width = array.shape
      rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

      for class_id, color in enumerate(palette):
        rgb_image[array == class_id] = color
      rgb_image[array == 255] = [255, 255, 255]
      
      image = Image.fromarray(rgb_image)
      image.save(save_path)
  return