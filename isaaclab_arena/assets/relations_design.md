This document should describe the required task to solve:

The question at hand is to find valid object palcement positions for multiple objects at the same time. I would like to start with a first version in 2d space as a proof
of concept. 


An asset can hold lists of relations. The relation needs to store the parent asset for later like so

class Relation:
    """Base class for spatial relationships between objects."""
    
    def __init__(self, child: Asset):
        self.child = child

 
class NextTo(Relation):
    """Represents a 'next to' spatial relationship."""
    
    def __init__(self, child: Asset, side: str = "right"):
        super().__init__(child)
        self.side = side
        print(f"[NextTo] Created: {self.child.name} will be placed {self.side} of parent")       


Then the user should be able to call relations via the following api

from isaaclab_arena.assets.relations import On, NextTo

packing_table = asset_registry.get_asset_by_name("packing_table")()

microwave.add_relation(On(packing_table))
cracker_box.add_relation(On(packing_table), NextTo(microwave))
apple.add_realation(NextTo(cracker_box))


