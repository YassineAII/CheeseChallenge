import os
import subprocess

def list_folders_in_directory(directory):
    try:
        entries = os.listdir(directory)
        folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        return folders
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied to access {directory}.")
        return []

directory_path = "/dataset/val"
folders = list_folders_in_directory(directory_path)


for x in folders:
    os.environ['MODEL_NAME'] = "CompVis/stable-diffusion-v1-4"
    os.environ['INSTANCE_DIR'] = f"dataset/finetune/{x}"
    os.environ['OUTPUT_DIR'] = f"generators/saves/{x}"

    command = [
        "accelerate", "launch", "train_dreambooth.py",
        "--pretrained_model_name_or_path", os.environ['MODEL_NAME'],
        "--instance_data_dir", os.environ['INSTANCE_DIR'],
        "--output_dir", os.environ['OUTPUT_DIR'],
        "--instance_prompt", f"{x} cheese".lower(),
        "--resolution", "512",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "5e-6",
        "--max_train_steps", "400"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print("Output:", result.stdout)
    print("Errors:", result.stderr)

