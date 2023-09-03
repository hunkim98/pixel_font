# pip install Pillow
from PIL import Image, ImageFont, ImageDraw
import os

# use a truetype font (.ttf)
# font file from fonts.google.com (https://fonts.google.com/specimen/Courier+Prime?query=courier)

def main():
    root_font_folder = "fonts"
    categories = [name for name in os.listdir(root_font_folder) if os.path.isdir(os.path.join(root_font_folder, name))]
    print("You have the following categories:")
    print(categories)

    for category in categories:
        category_font_path = f'{root_font_folder}/{category}'
        fonts = [name for name in os.listdir(category_font_path) if os.path.isdir(os.path.join(category_font_path, name))]
        for font in fonts:
            print(f'extracting png from {font}')
            ttf_file = [name for name in os.listdir(category_font_path +"/"+ font) if name.endswith('.ttf')]
            # continue
            if(len(ttf_file) == 0):
                print("No TTF file found in directory")
                continue
            print(ttf_file[0])
            font_name = ttf_file[0]
            out_path = "data/" + category + "/" + font + "/"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            # Find the maximum character width and height
            font_size = 25 # px this is to match the 28 x 28 MNIST dataset
            font_color = "#000000" # HEX Black

            # Copy Desired Characters from Google Fonts Page and Paste into variable
            desired_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789?!,."

            # A: 65
            # Z: 90
            # a: 97
            # z: 122

            # we set width and height to 28 to match the MNIST dataset
            image_size = 28
            font_path = category_font_path + "/" + font + "/" + font_name
            font_size = 1
            max_heights = [0]
            max_widths = [0]
            font = ImageFont.truetype(font_path, font_size)
            for char in desired_characters:
                char_height = font.getsize(char)[1] # [0] is width, [1] is height
                char_width = font.getsize(char)[0]
                max_widths.append(char_width)
                max_heights.append(char_height)
            while max(max_heights) < image_size and max(max_widths) < image_size:
                font_size += 1
                font = ImageFont.truetype(font_path, font_size)
                max_heights = [0]
                max_widths = [0]
                for char in desired_characters:
                    char_height = font.getsize(char)[1]
                    char_width = font.getsize(char)[0]
                    max_heights.append(char_height)
                    max_widths.append(char_width)
                    
            font_size -= 1

            # Loop through the characters and save them with the same font size
            for character in desired_characters:
                
                
                # Create a new image for each character
                char_img = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 0))
                char_draw = ImageDraw.Draw(char_img)
                
                # Calculate the size of the character
                char_width, char_height = char_draw.textsize(character, font=font)
                
                # Calculate the position to center the character both vertically and horizontally within the image
                x = (image_size - char_width) / 2
                y = (image_size - char_height) / 2
                
                # Draw the character with the same font size
                char_draw.text((x, y), str(character), font=ImageFont.truetype(font_path, font_size), fill=font_color)
                
                # Save the character as a PNG
                try:
                    char_img.save(out_path + str(ord(character)) + ".png")
                except:
                    print(f"[-] Couldn't Save:\t{character}")

if __name__ == "__main__":
    main()