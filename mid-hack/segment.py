import argparse
import os
import cv2
import torch
from fastsam import FastSAM, FastSAMPrompt

def get_device():
    """
    Returns the best available device (GPU or CPU).
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path, target_size=(512, 512)):
    """
    Reads and resizes an image to the target size.
    """
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image, target_size

def segment_image(model, image_path, device, img_size=1024, conf=0.4, iou=0.9):
    """
    Segments an image using the FastSAM model.
    """
    return model(
        source=image_path,
        device=device,
        retina_masks=True,
        imgsz=img_size,
        conf=conf,
        iou=iou
    )

def save_masks(masks, output_dir, original_image, device):
    """
    Saves individual masks and the annotated image with all masks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save individual masks
    for i, mask in enumerate(masks):
        mask_image = mask.cpu().numpy().astype("uint8") * 255
        mask_filename = os.path.join(output_dir, f"mask_{i}.png")
        cv2.imwrite(mask_filename, mask_image)
        print(f"Saved mask: {mask_filename}")

    # Save annotated image
    annotated_image_path = os.path.join(output_dir, "annotated_image.png")
    annotated_image = FastSAMPrompt.plot_masks(masks, original_image)
    cv2.imwrite(annotated_image_path, annotated_image)
    print(f"Saved annotated image: {annotated_image_path}")

def main(args):
    """
    Main function to segment an image and save the results.
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load FastSAM model
    model = FastSAM(args.checkpoint)

    # Segment image
    results = segment_image(model, args.image, device, img_size=args.img_size, conf=args.conf, iou=args.iou)

    # Generate masks
    prompt_processor = FastSAMPrompt(args.image, results, device=device)
    masks = prompt_processor.everything_prompt()

    # Save results
    original_image = cv2.imread(args.image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    save_masks(masks, args.output, original_image, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a retail image into product masks using FastSAM.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the FastSAM model checkpoint.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output", type=str, required=True, help="Path to save output masks and annotated image.")
    parser.add_argument("--img_size", type=int, default=1024, help="Image size for the model (default: 1024).")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4).")
    parser.add_argument("--iou", type=float, default=0.9, help="IoU threshold (default: 0.9).")

    args = parser.parse_args()
    main(args)


# run example
# python segment.py --checkpoint path_to_fast_sam_checkpoint.pth --image path_to_input_image.jpg --output output_directory
