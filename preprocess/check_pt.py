import torch
import os
from pathlib import Path

def check_single_pt_file(pt_path):
    """Check a single PT file for coordinates"""
    try:
        data = torch.load(pt_path, map_location='cpu')
        
        if isinstance(data, dict):
            keys = list(data.keys())
            has_coords = 'coords' in keys
            
            if has_coords:
                coords_shape = data['coords'].shape
                return True, keys, coords_shape
            else:
                return False, keys, None
        else:
            return False, "Not a dictionary", None
            
    except Exception as e:
        return False, f"Error: {e}", None

def check_pt_files_in_directory(directory):
    """Check all PT files in a directory"""
    pt_files = list(Path(directory).glob("*.pt"))
    
    if not pt_files:
        print(f"No PT files found in {directory}")
        return
    
    print(f"\nChecking {len(pt_files)} PT files in {directory}")
    print("="*60)
    
    with_coords = 0
    without_coords = 0
    
    for pt_file in pt_files[:10]:  # Check first 10 files
        has_coords, keys, coords_shape = check_single_pt_file(pt_file)
        
        if has_coords:
            print(f"✅ {pt_file.name}: Has coordinates {coords_shape}")
            with_coords += 1
        else:
            print(f"❌ {pt_file.name}: No coordinates | Keys: {keys}")
            without_coords += 1
    
    if len(pt_files) > 10:
        print(f"... and {len(pt_files) - 10} more files")
    
    print(f"\nSummary:")
    print(f"  With coordinates: {with_coords}")
    print(f"  Without coordinates: {without_coords}")

def quick_check_sample_files():
    """Quick check of sample files from each split"""
    
    # Directories to check
    directories = [
        "data/features_resnet18/train",
        "data/features_resnet18/val", 
        "data/features_resnet18/test"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            check_pt_files_in_directory(directory)
        else:
            print(f"Directory not found: {directory}")

def check_specific_files():
    """Check specific UCMC and BCR-NET files"""
    
    # Sample files to check
    sample_files = [
        "data/features_resnet18/train/UCH_BRCA_RS_1.pt",  # UCMC
        "data/features_resnet18/train/10.pt",             # BCR-NET
        "data/features_resnet18/train/11.pt",             # BCR-NET
    ]
    
    print("\nChecking specific sample files:")
    print("="*60)
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            has_coords, keys, coords_shape = check_single_pt_file(file_path)
            
            file_type = "UCMC" if "UCH_BRCA" in file_path else "BCR-NET"
            
            if has_coords:
                print(f"✅ {Path(file_path).name} ({file_type}): Has coordinates {coords_shape}")
                
                # Show sample coordinates
                data = torch.load(file_path, map_location='cpu')
                print(f"   Sample coords: {data['coords'][:3]}")
            else:
                print(f"❌ {Path(file_path).name} ({file_type}): No coordinates | Keys: {keys}")
        else:
            print(f"❓ {file_path}: File not found")

def detailed_analysis(pt_path):
    """Detailed analysis of a specific PT file"""
    print(f"\nDetailed analysis of: {pt_path}")
    print("="*60)
    
    if not os.path.exists(pt_path):
        print("File not found!")
        return
    
    try:
        data = torch.load(pt_path, map_location='cpu')
        
        print(f"File type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: {type(value)} - {value}")
            
            if 'coords' in data:
                coords = data['coords']
                print(f"\nCoordinate details:")
                print(f"  Shape: {coords.shape}")
                print(f"  Min: {coords.min()}")
                print(f"  Max: {coords.max()}")
                print(f"  First 5 coordinates:")
                print(f"    {coords[:5]}")
        
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    print("🔍 Checking PT files for coordinates...")
    
    # Quick overview
    quick_check_sample_files()
    
    # Check specific files
    check_specific_files()
    
    # Uncomment for detailed analysis of a specific file
    # detailed_analysis("data/features_resnet18/train/10.pt")