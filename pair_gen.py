import os

src = '/home/jsmoon/kaggle/input/image-matching-challenge-2023/train/heritage/cyprus'
img_dir = os.path.join(src, 'images')

imgs = os.listdir(img_dir)
imgs.sort()
img_pairs = list()
for i in range(len(imgs)):
    img_pairs.append([imgs[i], imgs[(i + 1) % len(imgs)]])
    img_pairs.append([imgs[i], imgs[(i + 2) % len(imgs)]])

print(imgs)
print(img_pairs)
print(len(img_pairs))
# Specify the output file name
output = '/home/jsmoon/kaggle/spsg/heritage_cyprus'
output_file = os.path.join(output, 'pairs-sfm.txt')

# Open the file for writing
with open(output_file, "w") as file:
    # Iterate through each pair in the list
    for pair in img_pairs:
        # Ensure that it's an n*2 list (each sub-list contains exactly 2 elements)
        if len(pair) == 2:
            # Write the pair as a line in the text file, separated by a space
            file.write(f"{pair[0]} {pair[1]}\n")
        else:
            print(
                f"Warning: Found a sub-list with length {len(pair)} instead of 2."
            )
            # You can choose to handle this case differently if needed