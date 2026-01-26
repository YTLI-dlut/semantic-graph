
import imageio.v2 as imageio
import os
import shutil
import sys

def make_gif(path):
    print(f"Processing {path}...")
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return

    file_names = sorted([fn for fn in os.listdir(path) if fn.endswith('.png')])
    
    if len(file_names) == 0:
        print("No images found.")
        return
        
    images = []
    for filename in file_names:
        images.append(imageio.imread(os.path.join(path, filename)))
        
    save_path = path + ".gif"
    print(f"Saving GIF to {save_path}")
    imageio.mimsave(save_path, images, duration=0.1)
    
    print("Deleting image folder...")
    shutil.rmtree(path)
    print("Done.")

if __name__ == "__main__":
    target_dir = "/home/iiau/createGraph_ws/gifs/semantic_20260123_152642/episode_0"
    make_gif(target_dir)
