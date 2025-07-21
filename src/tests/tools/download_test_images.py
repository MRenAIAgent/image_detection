#!/usr/bin/env python3
"""
Download test images for comprehensive API testing
This script downloads sample images with various objects for testing the Image Recognition API
"""

import os
import requests
from pathlib import Path
import time

# Test images to download
TEST_IMAGES = {
    "cars_street.jpg": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=640&h=480&fit=crop&crop=center",
    "dog_park.jpg": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=640&h=480&fit=crop&crop=center", 
    "coffee_table.jpg": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=640&h=480&fit=crop",
    "kitchen_scene.jpg": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&h=480&fit=crop",
    "living_room.jpg": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=640&h=480&fit=crop",
    "person_bicycle.jpg": "https://images.unsplash.com/photo-1571068316344-75bc76f77890?w=640&h=480&fit=crop"
}

def download_image(filename: str, url: str, output_dir: Path) -> bool:
    """Download a single image."""
    output_path = output_dir / filename
    
    try:
        print(f"📥 Downloading {filename}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        file_size = output_path.stat().st_size
        print(f"✅ Downloaded {filename} ({file_size:,} bytes)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def main():
    """Download all test images."""
    print("🖼️  Image Recognition API - Test Image Downloader")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("tests/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Check if images already exist
    existing_images = []
    for filename in TEST_IMAGES.keys():
        if (output_dir / filename).exists():
            existing_images.append(filename)
    
    if existing_images:
        print(f"\n✅ Found {len(existing_images)} existing images:")
        for img in existing_images:
            file_size = (output_dir / img).stat().st_size
            print(f"   - {img} ({file_size:,} bytes)")
        
        # Only download missing images
        images_to_download = {k: v for k, v in TEST_IMAGES.items() 
                            if k not in existing_images}
        
        if not images_to_download:
            print(f"\n✅ All {len(TEST_IMAGES)} test images already exist!")
            print(f"📁 Location: {output_dir.absolute()}")
            print(f"\n🧪 Run tests with: python3 tests/test_local_with_images.py")
            return
        else:
            print(f"\n📥 Will download {len(images_to_download)} missing images...")
    else:
        images_to_download = TEST_IMAGES
    
    # Download images
    print(f"\n📥 Downloading {len(images_to_download)} images...")
    print("-" * 40)
    
    successful = 0
    failed = 0
    
    for filename, url in images_to_download.items():
        if download_image(filename, url, output_dir):
            successful += 1
        else:
            failed += 1
        time.sleep(0.5)  # Be nice to the server
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Download Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Location: {output_dir.absolute()}")
    
    if successful > 0:
        print(f"\n🧪 Test your API with:")
        print(f"   python3 tests/test_local_with_images.py")
        print(f"\n📖 View test image details:")
        print(f"   cat tests/data/README.md")
    
    if failed > 0:
        print(f"\n⚠️  {failed} images failed to download. You can:")
        print("   1. Run this script again")
        print("   2. Download images manually to tests/data/")
        print("   3. Use your own test images")

if __name__ == "__main__":
    main() 