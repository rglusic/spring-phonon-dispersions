from cell_structure import CellStructure
from render_cell import RenderCell
import numpy as np
import vpython as vp

def spring_solve(cell, initial_displacement, threshold):
    renderer = RenderCell(cell, threshold)
    renderer.plot_supplimentary_information()
    renderer.menu()
    
    renderer.change_atom_pos(initial_displacement)
    
    renderer.render()

def generate_fcc(cell, a, num_of_atoms=100):
    # Outer corners (back)
    cell.place_atom('Si', vp.vector(a/2, a/2., a/2))
    cell.place_atom('Si', vp.vector(a/2, -a/2.,a/2))
    cell.place_atom('Si', vp.vector(-a/2, a/2, a/2))
    cell.place_atom('Si', vp.vector(-a/2,-a/2,a/2))
    
    # Outer corners (front)
    cell.place_atom('Si', vp.vector(a/2,a/2,-a/2))
    cell.place_atom('Si', vp.vector(a/2,-a/2,-a/2))
    cell.place_atom('Si', vp.vector(-a/2,a/2,-a/2))
    cell.place_atom('Si', vp.vector(-a/2,-a/2,-a/2))
    
    # Faces and center atom
    cell.place_atom('Si', vp.vector(a/2,0., 0.))
    cell.place_atom('Si', vp.vector(-a/2,0., 0.))
    cell.place_atom('Si', vp.vector(0.,0.,-a/2))
    cell.place_atom('Si', vp.vector(0.,0.,0.))
    cell.place_atom('Si', vp.vector(0.,0.,a/2))

# Run program
a = 3.57 #angstroms

# Useless for now.
cell = [[a/2, 0, 0],
        [0, a/2, 0],
        [0, 0, a/2]]

cell_structure = CellStructure(np.array(cell))

generate_fcc(cell_structure, a)

spring_solve(cell_structure, vp.vector(0.5,0,0), a)