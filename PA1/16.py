import pandas as pd
import numpy as np
from PIL import Image
image_path = "/run/media/vector/New Volume/ML/PA1/image.jpg"
image = Image.open(image_path)
image_array = np.array(image)
if len(image_array.shape) == 3:
    image_array = image_array.reshape(-1, image_array.shape[-1])
else:
    image_array = image_array.reshape(-1, 1)
df = pd.DataFrame(image_array)
output_path = "/run/media/vector/New Volume/ML/PA1/output.csv"
df.to_csv(output_path, index=False)