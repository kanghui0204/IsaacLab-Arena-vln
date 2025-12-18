# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets.asset import Asset


class Relation:
    """Base class for spatial relationships between objects."""
    
    def __init__(self, child: Asset):
        self.child = child


class On(Relation):
    """Represents an 'on top of' spatial relationship."""
    
    def __init__(self, child: Asset):
        super().__init__(child)
        print(f"[On] Created: {self.child.name} will be placed on top of parent")


class NextTo(Relation):
    """Represents a 'next to' spatial relationship."""
    
    def __init__(self, child: Asset, side: str = "right"):
        super().__init__(child)
        self.side = side
        print(f"[NextTo] Created: {self.child.name} will be placed {self.side} of parent")