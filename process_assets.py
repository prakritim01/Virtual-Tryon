import os
from PIL import Image

def remove_white_background(folder_path="assets"):
    print(f"🛠️ Starting Asset Pipeline: Processing images in '{folder_path}'...")
    
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Could not find '{folder_path}' folder.")
        return

    # Loop through every image in the assets folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            
            # Open the image and convert it to an RGBA format (Red, Green, Blue, Alpha/Transparency)
            img = Image.open(file_path).convert("RGBA")
            datas = img.getdata()

            new_data = []
            for item in datas:
                # Check if the pixel is white (or very light gray). 
                # R, G, and B values above 230 are considered "white"
                if item[0] > 230 and item[1] > 230 and item[2] > 230:
                    # Replace with a completely transparent pixel
                    new_data.append((255, 255, 255, 0))
                else:
                    # Keep the original pixel
                    new_data.append(item)

            # Apply the new transparent data to the image
            img.putdata(new_data)
            
            # Overwrite the old file with the new transparent one
            img.save(file_path, "PNG")
            print(f"  ✅ Successfully made transparent: {filename}")

if __name__ == "__main__":
    remove_white_background()
    print("🎉 Asset processing complete! You can now run MaisonMuse.")