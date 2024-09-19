# src/api.py
import os
import requests

def get_images(limit=5):
    """Fetches the first `limit` number of images from the ISIC API (v2)."""
    url = f"https://api.isic-archive.com/api/v2/images/?limit={limit}"
    headers = {'Accept': 'application/json'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

def download_image(image_url, save_path):
    """Downloads an image from a URL and saves it to the given path."""
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded image to {save_path}")
    else:
        print(f"Failed to download image from {image_url}, status code: {response.status_code}")

def fetch_and_save_images(limit=5, image_dir='images'):
    """Fetches images from the API and downloads them to the `image_dir`."""
    os.makedirs(image_dir, exist_ok=True)
    images = get_images(limit=limit)
    image_filenames = []
    label_list = []
    
    if images:
        for img in images['results']:
            image_id = img['isic_id']
            image_url = img['files']['full']['url']  # Full image URL
            save_path = f"{image_dir}/{image_id}.jpg"
            download_image(image_url, save_path)
            image_filenames.append(image_id)
            
            # Example: Set binary labels based on 'benign_malignant' metadata
            label_list.append([1, 0] if img['metadata']['clinical']['benign_malignant'] == 'benign' else [0, 1])
    
    return image_filenames, label_list
