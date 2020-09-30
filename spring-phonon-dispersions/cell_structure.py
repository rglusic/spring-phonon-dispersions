import numpy as np
import vpython as vp

class CellStructure(object):
    """ Controls the structure of the 
    atomic cell structure we're examining.
    """
    cell = np.array([])
    atom_list = []
    force_dict = {
        'C': {'force_constant': 1.0}
    }
    
    def __init__(self, cell):
        self.cell = cell
    
    @property
    def get_cell(self):
        return self.cell
    
    def place_atom(self, atom_name, position):
        # See if we have a force constant for the particle
        try:
            force = self.force_dict.get(atom_name).get('force_constant')
        except KeyError:
            print("Force constant not found for '{}', assuming one.".format(atom_name))
            force = 1.0
        
        # Append the new atom to the atom list
        self.atom_list.append((atom_name, position, force))
        