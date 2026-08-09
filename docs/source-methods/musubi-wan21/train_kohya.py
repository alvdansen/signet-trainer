import os
from pathlib import Path

from modal import (
    App,
    Image,
    Secret,
    Volume,
)
app = App(name="kohya-trainer")
dataset_volume = Volume.from_name("datasets", create_if_missing=True)

@app.local_entrypoint()
def add_to_volume_datasets(timeout=3600):
    FILES_TO_ADD_DATASETS = [
        {"source": f"datasets", "destination": "/datasets"},
    ]
    with dataset_volume.batch_upload() as batch:
        for file in FILES_TO_ADD_DATASETS:
            batch.put_directory(file["source"], file["destination"])
    return "Success"

image = (
    Image.debian_slim(python_version="3.10")
    .env({"HF_HUB_CACHE": "/cache/cache/"})
    .apt_install("git")
    .apt_install("ffmpeg")
    .apt_install("python3-opencv")
    .pip_install("transformers>=4.46.0")
    .pip_install("huggingface_hub")
    .run_commands("git clone https://github.com/kohya-ss/musubi-tuner.git")
    .workdir("musubi-tuner")
    .run_commands(
        "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        "pip3 install -r requirements.txt",
    )
    .pip_install("pydantic==1.10.13", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install("albumentations==1.4.3")
    .add_local_file(
        "wan21-dataset-config.toml", 
        "/dataset-config.toml"
    )
    # .add_local_dir(
    #     "datasets/sapo_concho_test/", 
    #     "/datasets"
    # )
)

volume = Volume.from_name("kohya-volume", create_if_missing=True)
cache_vol = Volume.from_name("hf-hub-cache", create_if_missing=True)
OUTPUT_DIR = "/outputs"

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={OUTPUT_DIR: volume, "/cache": cache_vol, "/datasets": dataset_volume},
    timeout=60 * 60 * 24,
    secrets=[Secret.from_name("my-huggingface-secret")],
)
def train():
    import subprocess
    import os
    import glob
    import huggingface_hub
    from pathlib import Path

    # Model repositories
    i2v_repo_id = "Wan-AI/Wan2.1-I2V-14B-720P"
    comfy_repo_id = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
    
    try:
        t5_path = huggingface_hub.hf_hub_download(
            repo_id=i2v_repo_id, 
            filename="models_t5_umt5-xxl-enc-bf16.pth",
            local_files_only=False
        )
        clip_path = huggingface_hub.hf_hub_download(
            repo_id=i2v_repo_id, 
            filename="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            local_files_only=False
        )
        vae_path = huggingface_hub.hf_hub_download(
            repo_id=comfy_repo_id, 
            filename="split_files/vae/wan_2.1_vae.safetensors",
            local_files_only=False
        )
        dit_path = huggingface_hub.hf_hub_download(
            repo_id=comfy_repo_id, 
            filename="split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors",
            local_files_only=False
        )
        
        print(f"Found cached models in Hugging Face cache:")
        print(f"DiT model: {dit_path}")
        print(f"VAE model: {vae_path}")
        print(f"T5 model: {t5_path}")
        print(f"CLIP model: {clip_path}")
    except Exception as e:
        print(f"Error finding cached model files: {e}")
        print("Please ensure the model files are properly cached before running this script.")
        return

    print("Step 1: Caching latents for WAN model...")
    subprocess.run(
        [
            "python",
            "wan_cache_latents.py",
            "--dataset_config", "/dataset-config.toml",
            "--vae", vae_path,
            "--clip", clip_path,
            "--vae_cache_cpu"
        ]
    )
    
    print("Step 2: Caching text encoder outputs for WAN model...")
    subprocess.run(
        [
            "python",
            "wan_cache_text_encoder_outputs.py",
            "--dataset_config", "/dataset-config.toml",
            "--t5", t5_path,
            "--batch_size", "16",
            "--fp8_t5"
        ]
    )

    print("Step 3: Starting WAN model training...")
    subprocess.run(
        [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "1",
            "--mixed_precision", "bf16",
            "wan_train_network.py",
            "--task", "t2v-14B",
            "--dit", dit_path,
            "--vae", vae_path,
            "--t5", t5_path,
            "--clip", clip_path, 
            "--dataset_config", "/dataset-config.toml",
            "--sdpa",
            "--mixed_precision", "bf16",
            "--fp8_base",
            "--fp8_scaled",
            "--vae_cache_cpu",
            "--optimizer_type", "adamw8bit",
            "--optimizer_args", "weight_decay=0.01",
            "--learning_rate", "2e-4",
            "--gradient_checkpointing",
            "--max_data_loader_n_workers", "2",
            "--persistent_data_loader_workers",
            "--network_module", "networks.lora_wan",
            "--network_dim", "32",
            "--discrete_flow_shift", "7.0",
            "--max_train_epochs", "16",
            "--save_every_n_epochs", "1",
            "--seed", "42",
            "--output_dir", OUTPUT_DIR,
            "--output_name", "wan21-lora"
        ]
    )
    print(f"Training complete. All files have been saved to the Modal volume in {OUTPUT_DIR}")
    print(f"Contents of {OUTPUT_DIR}:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        level = root.replace(OUTPUT_DIR, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

    volume.commit()

@app.local_entrypoint()
def run():
    train.remote()