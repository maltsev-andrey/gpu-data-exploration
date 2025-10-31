#!/usr/bin/env python3

"""
HDF5 Dataset Explorer
Analyzes structure and content of GBSense HDF5 files
"""

import os
import sys
import h5py
import numpy as np

# Check if user roor or not
if os.getuid() == 0:
    print("\n"+"-"*60 )
    print("ERROR: Don't run GPU as root! Use: su - ansible")
    print("-"*60 + "\n")
    sys.exit(1)
    
# Patch NumPy for scikit-cuda compatibility
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'complex'):
    np.complex = np.complex128

def explore_h5_file(filepath):
    """Recursively explore HDF5 file structure"""

    def print_structure(name, obj):
        """Print HDF5 object structure"""
        if isinstance(obj, h5py.Dataset):
            print(f"\nDATASET: {name}")
            print(f"  Shape: {obj.shape}")
            print(f"  Dtype: {obj.dtype}")
            print(f"  Size: {obj.size:,} elements")

            # Calculate memory size
            bytes_size = obj.size * obj.dtype.itemsize
            if bytes_size > 1e9:
                size_str = f"{bytes_size / 1e9:.2f} GB"
            elif bytes_size > 1e6:
                size_str = f"{bytes_size / 1e6:.2f} MB"
            elif bytes_size > 1e3:
                size_str = f"{bytes_size / 1e3:.2f} KB"
            else:
                size_str = f"{bytes_size} bytes"
            print(f"  Memory: {size_str}")

            # Show sample data for small dataset
            if obj.size > 0 and obj.size <= 100:
                print(f"  Data:\n{obj[:]}")
            elif obj.size > 0:
                # Show shape and sample values
                if len(obj.shape) == 1:
                    sample = obj[:min(10, obj.shape[0])]
                    print(f"  Sample (first {len(sample)}): {sample}")
                    if obj.shape[0] > 10:
                        sample_data = obj[:1000]
                        print(f"  Stats: min={np.min(sample_data)}, "
                              f"max={np.max(sample_data)}, "
                              f"mean={np.mean(sample_data):.6f}")
                elif len(obj.shape) == 2:
                    rows = min(3, obj.shape[0])
                    cols = min(5, obj.shape[1])
                    sample = obj[:rows, :cols]
                    print(f"  Sample ({rows}x{cols}):\n{sample}")
                    if obj.size > 1000:
                        chunk = obj[:min(100, obj.shape[0])]
                        print(f"  Stats: min={np.min(chunk)}, "
                          f"max={np.max(chunk)}, "
                          f"mean={np.mean(chunk):.6f}")
                elif len(obj.shape) == 3:
                    # 3D array - show first slice
                    d0, d1, d2 = min(2, obj.shape[0]), min(5, obj.shape[1]), min(5, obj.shape[2])
                    sample = obj[:d0, :d1, :d2]
                    print(f"  Sample ({d0}x{d1}x{d2}):")
                    print(f"    First slice:\n{sample[0]}")
                    if obj.size > 1000:
                        chunk = obj[:min(10, obj.shape[0])]
                        print(f"  Stats: min={np.min(chunk)}, "
                              f"max={np.max(chunk)}, "
                              f"mean={np.mean(chunk):.6f}")
                else:
                    print(f"  Multi-dimensional array (showing first element):")
                    idx = tuple([0] * len(obj.shape))
                    print(f"  {obj[idx]}")
                    
        elif isinstance(obj, h5py.Group):
            print(f"\nGROUP: {name}")
            print(f"  Keys: {list(obj.keys())}")

    try:
        print("="*80)
        print(f"ANALYZING: {filepath}")
        print("="*80)

        with h5py.File(filepath, 'r') as f:
            print(f"\nRoot keys: {list(f.keys())}")
            print("\nFull structure:")
            print("-" * 80)
            f.visititems(print_structure)

        print("\n" + "="*80)
        print("EXPLORATION COMPLETE")
        print("="*80)

    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_h5.py <h5_file>")
        print("\nExample:")
        print("  python explore_h5.py data_1/data_1_train.h5")
        sys.exit(1)
                        
    filepath = sys.argv[1]
    explore_h5_file(filepath)


























