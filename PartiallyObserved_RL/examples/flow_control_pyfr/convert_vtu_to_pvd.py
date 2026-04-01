#!/usr/bin/env python3
"""
Create PVD collection file from existing VTU files for ParaView animation
"""

from pathlib import Path
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_pvd_from_vtu_files(vtu_directory, output_name='animation.pvd'):
    """
    Create a PVD file from existing VTU files
    
    Args:
        vtu_directory: Directory containing .vtu files
        output_name: Name for output PVD file
    """
    vtu_dir = Path(vtu_directory)
    
    if not vtu_dir.exists():
        print(f"❌ Error: Directory not found: {vtu_dir}")
        return False
    
    # Find all VTU files
    vtu_files = sorted(vtu_dir.glob('*.vtu'))
    
    if not vtu_files:
        print(f"❌ No VTU files found in {vtu_dir}")
        return False
    
    print("="*70)
    print("PVD COLLECTION GENERATOR")
    print("="*70)
    print(f"Directory: {vtu_dir}")
    print(f"Found {len(vtu_files)} VTU files")
    print()
    
    # Create XML structure
    root = ET.Element('VTKFile', {
        'type': 'Collection',
        'version': '0.1',
        'byte_order': 'LittleEndian'
    })
    
    collection = ET.SubElement(root, 'Collection')
    
    # Extract time values and add to collection
    time_vtu_pairs = []
    
    for vtu_file in vtu_files:
        # Try to extract time from filename: flow_t75.50.vtu -> 75.50
        match = re.search(r't(\d+\.?\d*)\.vtu$', vtu_file.name)
        if match:
            time = float(match.group(1))
        else:
            # Fallback: use alphabetical order
            time = float(len(time_vtu_pairs))
        
        time_vtu_pairs.append((time, vtu_file))
    
    # Sort by time
    time_vtu_pairs.sort(key=lambda x: x[0])
    
    # Add each file to collection
    for time, vtu_file in time_vtu_pairs:
        ET.SubElement(collection, 'DataSet', {
            'timestep': str(time),
            'group': '',
            'part': '0',
            'file': vtu_file.name  # Use relative path
        })
    
    # Pretty print XML
    xml_string = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(xml_string)
    pretty_xml = dom.toprettyxml(indent='  ')
    
    # Remove extra blank lines
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    # Write to file
    output_path = vtu_dir / output_name
    with open(output_path, 'w') as f:
        f.write(pretty_xml)
    
    print(f"✓ Created PVD file: {output_path}")
    print(f"  Timesteps: {len(time_vtu_pairs)}")
    print(f"  Time range: {time_vtu_pairs[0][0]:.2f} - {time_vtu_pairs[-1][0]:.2f}")
    print()
    print("="*70)
    print("HOW TO USE IN PARAVIEW")
    print("="*70)
    print()
    print("1. Open ParaView")
    print()
    print("2. File → Open → Select:")
    print(f"   {output_path}")
    print()
    print("3. Click 'Apply'")
    print()
    print("4. Use the Play button ▶ to animate!")
    print()
    print("That's it! The PVD file links all VTU files as a time series.")
    print("="*70)
    
    return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create PVD collection from VTU files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create PVD in the vtu_files directory
  python %(prog)s /home/aayushman/.../solution_files/vtu_files
  
  # Custom output name
  python %(prog)s /path/to/vtu/files --output my_animation.pvd
        '''
    )
    
    parser.add_argument(
        'vtu_directory',
        nargs='?',
        default='/home/aayushman/PartiallyObserved_RL/examples/flow_control_pyfr/Flow_Experiments/exp_re200/solutions/vtu_files',
        help='Directory containing VTU files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='animation.pvd',
        help='Output PVD filename'
    )
    
    args = parser.parse_args()
    
    success = create_pvd_from_vtu_files(args.vtu_directory, args.output)
    
    import sys
    sys.exit(0 if success else 1)