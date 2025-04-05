import sys
from PIL import Image
import argparse

def gray2mono(gray_image_path, binary_image_path, r, threshold):
    try:
        img = Image.open(gray_image_path).convert('L')
        width, height = img.size
        pixels = img.load()

        binary_img = Image.new('1', (width, height))
        binary_pixels = binary_img.load()

        for i in range(height):
            for j in range(width):
                neighbor_sum = 0
                neighbor_count = 0

                for row in range(max(0, i - r), min(height, i + r + 1)):
                    for col in range(max(0, j - r), min(width, j + r + 1)):
                        neighbor_sum += pixels[col, row]
                        neighbor_count += 1

                mean = neighbor_sum / neighbor_count if neighbor_count > 0 else 0

                if mean <= threshold:
                    binary_pixels[j, i] = 0
                else:
                    binary_pixels[j, i] = 1

        binary_img.save(binary_image_path)
        print(f"Successfully converted '{gray_image_path}' to '{binary_image_path}' with r={r} and t={threshold}")

    except FileNotFoundError:
        print(f"Error: Input file '{gray_image_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert a grayscale image to a binary image")
    parser.add_argument("gray_image_path", type=str)
    parser.add_argument("binary_image_path", type=str)
    parser.add_argument("-r", "--radius", type=int, required=True)
    parser.add_argument("-t", "--threshold", type=int, required=True)

    args = parser.parse_args()

    gray2mono(args.gray_image_path, args.binary_image_path, args.radius, args.threshold)
