from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import os


def apply_blur_filter(image_path, output_path):
    try:
        img = Image.open(image_path)

        img_resized = img.resize((128, 128))

        img_blurred = img_resized.filter(
            ImageFilter.GaussianBlur(radius=2)
        )

        plt.imshow(img_blurred)
        plt.axis("off")
        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close()

        print(
            f"Blurred image saved as '{output_path}'."
        )

    except Exception as e:
        print(f"Error processing blur: {e}")


def apply_flameborne_filter(image_path, output_path):
    try:
        img = Image.open(image_path).convert("RGB")

        # Sharpen details
        sharpened = img.filter(
            ImageFilter.UnsharpMask(
                radius=2,
                percent=180,
                threshold=3
            )
        )

        # Increase contrast
        contrast = ImageEnhance.Contrast(
            sharpened
        ).enhance(1.30)

        # Increase color intensity
        colorful = ImageEnhance.Color(
            contrast
        ).enhance(1.40)

        # Create a grayscale copy for warm colorization
        gray = ImageOps.grayscale(colorful)

        # Add dark ember shadows and bright golden highlights
        warm_layer = ImageOps.colorize(
            gray,
            black="#2b0800",
            white="#ffd45c"
        )

        # Blend the warm effect with the original colors
        final_image = Image.blend(
            colorful,
            warm_layer,
            0.30
        )

        final_image.save(output_path)

        print(
            f"FlameBorne image saved as '{output_path}'."
        )

    except Exception as e:
        print(f"Error processing FlameBorne filter: {e}")


if __name__ == "__main__":
    print(
        "Image Filter Processor "
        "(type 'exit' to quit)\n"
    )

    while True:
        image_path = input(
            "Enter image filename "
            "(or 'exit' to quit): "
        ).strip()

        if image_path.lower() == "exit":
            print("Goodbye!")
            break

        if not os.path.isfile(image_path):
            print(
                f"File not found: {image_path}"
            )
            continue

        base, ext = os.path.splitext(image_path)

        blurred_output = (
            f"{base}_blurred{ext}"
        )

        flameborne_output = (
            f"{base}_flameborne{ext}"
        )

        apply_blur_filter(
            image_path,
            blurred_output
        )

        apply_flameborne_filter(
            image_path,
            flameborne_output
        )