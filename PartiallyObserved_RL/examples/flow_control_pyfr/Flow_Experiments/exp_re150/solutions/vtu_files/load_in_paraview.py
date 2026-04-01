# ParaView Python script to load flow data
# Run in ParaView: Tools → Python Shell → Run Script

from paraview.simple import *

# Load VTU files
reader = LegacyVTKReader(FileNames=[
    '/home/aayushman/PartiallyObserved_RL/examples/flow_control_pyfr/Flow_Experiments/exp_re150/solutions/vtu_files/flow_t*.vtu'
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
