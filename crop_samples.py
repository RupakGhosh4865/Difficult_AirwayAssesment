from PIL import Image
import os

base_dir = r"D:\difficult airway assesment\Difficult_AirwayAssesment"
public_dir = os.path.join(base_dir, "frontend", "public", "test-images")
diff_file = os.path.join(base_dir, "data", "difficult", "diff1.jpg")

if not os.path.exists(public_dir):
    os.makedirs(public_dir)

img = Image.open(diff_file)
w, h = img.size
# Assuming it's a 3x1 collage
part_w = w // 3

parts = [
    (0, 0, part_w, h),
    (part_w, 0, 2 * part_w, h),
    (2 * part_w, 0, w, h)
]

labels = ["neutral_2", "tongue_2", "headup_2"]

for i, box in enumerate(parts):
    part_img = img.crop(box)
    part_img.save(os.path.join(public_dir, f"{labels[i]}.jpg"))

print("Cropped diff1.jpg into 3 parts.")
