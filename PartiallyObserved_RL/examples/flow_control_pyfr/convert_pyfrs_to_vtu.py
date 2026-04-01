#!/usr/bin/env python3
"""
Convert all PyFR solution files (.pyfrs) to VTU format for ParaView visualization
"""

import subprocess
from pathlib import Path
import sys
import re

def convert_pyfrs_to_vtu(solution_dir, mesh_file, output_subdir='vtu_files'):
    """
    Convert all .pyfrs files in a directory to .vtu format
    
    Args:
        solution_dir: Directory containing .pyfrs files
        mesh_file: Path to .pyfrm mesh file
        output_subdir: Subdirectory name for VTU files (created inside solution_dir)
    """
    solution_dir = Path(solution_dir)
    mesh_file = Path(mesh_file)
    
    # Check inputs
    if not solution_dir.exists():
        print(f"❌ Error: Solution directory not found: {solution_dir}")
        return False
    
    if not mesh_file.exists():
        print(f"❌ Error: Mesh file not found: {mesh_file}")
        return False
    
    # Create output directory
    output_dir = solution_dir / output_subdir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all .pyfrs files
    pyfrs_files = sorted(solution_dir.glob('*.pyfrs'))
    
    if not pyfrs_files:
        print(f"❌ No .pyfrs files found in {solution_dir}")
        return False
    
    print("="*70)
    print("PYFR TO VTU CONVERTER")
    print("="*70)
    print(f"Solution directory: {solution_dir}")
    print(f"Mesh file: {mesh_file}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(pyfrs_files)} solution files")
    print("="*70)
    print()
    
    # Convert each file
    successful = 0
    failed = 0
    failed_files = []
    
    for i, pyfrs_file in enumerate(pyfrs_files, 1):
        # Extract time from filename
        # Matches patterns like: inc-cylinder-ptb-75.50.pyfrs -> 75.50
        match = re.search(r'(\d+\.\d+)\.pyfrs$', pyfrs_file.name)
        if match:
            time_str = match.group(1)
        else:
            # Fallback: use index
            time_str = f"{i:04d}"
        
        # Output filename
        vtu_file = output_dir / f'flow_t{time_str}.vtu'
        
        # Progress
        percent = (i / len(pyfrs_files)) * 100
        print(f"[{i:3d}/{len(pyfrs_files)}] ({percent:5.1f}%) Converting {pyfrs_file.name}...", end='')
        
        # Run pyfr export
        cmd = [
            'pyfr', 'export',
            str(mesh_file),
            str(pyfrs_file),
            str(vtu_file)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per file
            )
            
            if result.returncode == 0:
                successful += 1
                file_size = vtu_file.stat().st_size / (1024 * 1024)  # MB
                print(f" ✓ ({file_size:.1f} MB)")
            else:
                failed += 1
                failed_files.append((pyfrs_file.name, result.stderr[:100]))
                print(f" ✗")
                print(f"    Error: {result.stderr[:100]}")
                
        except subprocess.TimeoutExpired:
            failed += 1
            failed_files.append((pyfrs_file.name, "Timeout"))
            print(f" ✗ (Timeout)")
            
        except Exception as e:
            failed += 1
            failed_files.append((pyfrs_file.name, str(e)))
            print(f" ✗")
            print(f"    Error: {e}")
    
    # Summary
    print()
    print("="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"Total files:     {len(pyfrs_files)}")
    print(f"✓ Successful:    {successful}")
    print(f"✗ Failed:        {failed}")
    print()
    
    if failed > 0:
        print("Failed conversions:")
        for fname, error in failed_files:
            print(f"  • {fname}: {error}")
        print()
    
    if successful > 0:
        # Calculate total size
        total_size = sum(f.stat().st_size for f in output_dir.glob('*.vtu'))
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"VTU files saved to: {output_dir}")
        print(f"Total size: {total_size_mb:.1f} MB")
        print()
        print("="*70)
        print("HOW TO VISUALIZE IN PARAVIEW")
        print("="*70)
        print()
        print("1. Open ParaView")
        print()
        print("2. File → Open → Navigate to:")
        print(f"   {output_dir}")
        print()
        print("3. Select all flow_t*.vtu files (or use 'flow_t..vtu' pattern)")
        print()
        print("4. Click 'Apply' in the Properties panel")
        print()
        print("5. In the top toolbar:")
        print("   • Select coloring: 'p' (pressure) or 'U' (velocity)")
        print("   • Use time controls to animate")
        print()
        print("6. Recommended filters:")
        print("   • Glyph: Show velocity vectors")
        print("   • Stream Tracer: Show streamlines")
        print("   • Contour: Iso-surfaces")
        print()
        print("7. For vortex visualization:")
        print("   • Color by pressure")
        print("   • Add Contour filter")
        print("   • Set isosurface value to mean pressure")
        print("   • Animate to see vortex shedding")
        print()
        print("="*70)
        
        # Create a simple script for ParaView
        create_paraview_script(output_dir)
    
    return successful > 0

def create_paraview_script(output_dir):
    """Create a Python script to load data in ParaView"""
    script_path = output_dir / 'load_in_paraview.py'
    
    script_content = f'''# ParaView Python script to load flow data
# Run in ParaView: Tools → Python Shell → Run Script

from paraview.simple import *

# Load VTU files
reader = LegacyVTKReader(FileNames=[
    '{output_dir}/flow_t*.vtu'
])

# Show in render view
Show(reader)

# Color by pressure
ColorBy(reader, ('POINTS', 'p'))

# Reset camera
ResetCamera()

# Get color transfer function
pLUT = GetColorTransferFunction('p')

# Rescale to data range
reader.UpdatePipeline()
pLUT.RescaleTransferFunction(reader.PointData['p'].GetRange())

# Render
Render()

print("Data loaded! Use the time controls to animate.")
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Created ParaView script: {script_path}")
    print("  (Run this in ParaView: Tools → Python Shell → Run Script)")
    print()

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert PyFR solution files to VTU format for ParaView',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert all files in solution_files directory
  python %(prog)s
  
  # Specify custom paths
  python %(prog)s --solution-dir /path/to/solutions --mesh /path/to/mesh.pyfrm
  
  # Custom output directory name
  python %(prog)s --output-subdir my_vtu_files
        '''
    )
    
    parser.add_argument(
        '--solution-dir',
        type=str,
        default='/home/aayushman/PartiallyObserved_RL/examples/flow_control_pyfr/Flow_Experiments/exp_re200/solutions',
        help='Directory containing .pyfrs solution files'
    )
    
    parser.add_argument(
        '--mesh',
        type=str,
        default='/home/aayushman/PyFR-Flow-Control/2d-inc-cylinder-base/inc-cylinder.pyfrm',
        help='Path to .pyfrm mesh file'
    )
    
    parser.add_argument(
        '--output-subdir',
        type=str,
        default='vtu_files',
        help='Subdirectory name for VTU output (created inside solution-dir)'
    )
    
    args = parser.parse_args()
    
    # Run conversion
    success = convert_pyfrs_to_vtu(
        args.solution_dir,
        args.mesh,
        args.output_subdir
    )
    
    # Exit code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
