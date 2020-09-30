from cell_structure import CellStructure
from render_cell import RenderCell
import numpy as np
import vpython as vp

def spring_solve(cell):
    renderer = RenderCell(cell)
    renderer.plot_supplimentary_information()
    renderer.menu()
    renderer.render()

# Run program
a = 3.57 #angstroms
# Useless for now.
cell = [[a/2, 0, 0],
        [0, a/2, 0],
        [0, 0, a/2]]

cell_structure = CellStructure(np.array(cell))
cell_structure.place_atom('C', vp.vector(a/2,0.,0.))
cell_structure.place_atom('C', vp.vector(-a/2,0.,0.))
spring_solve(cell_structure)