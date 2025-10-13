Scene Design
============

Scenes manage collections of assets that define the physical environment for simulation. They provide a unified interface for composing backgrounds, objects, and interactive elements into complete environments.

Core Architecture
-----------------

Scenes use the ``Scene`` class that manages asset collections:

.. code-block:: python

   class Scene:
       def __init__(self, assets: list[Asset] | None = None):
           self.assets: dict[str, Asset] = {}

       def add_asset(self, asset: Asset):
           """Add single asset to scene."""
           self.assets[asset.name] = asset

       def get_scene_cfg(self) -> Any:
           """Generate Isaac Lab scene configuration from assets."""

Scenes automatically aggregate asset configurations into Isaac Lab-compatible scene configurations while maintaining asset relationships and spatial organization.

Scenes in Detail
----------------

**Asset Composition**
   Scenes organize different asset types into cohesive environments:

   - **Background Assets**: Environmental foundations (kitchens, laboratories) providing base geometry and context
   - **Interactive Objects**: Manipulable items with physics (YCB objects, tools, appliances)
   - **Functional Elements**: Objects with affordances (doors, drawers, buttons) for specific interactions
   - **Object References**: Access to embedded scene elements without separate spawning

**Configuration Management**
   Automatic aggregation of asset configurations:

   - **Scene Configuration**: Physical properties, lighting, materials from all assets
   - **Pose Management**: Initial positioning and spatial relationships

Environment Integration
-----------------------

.. code-block:: python

   # Asset creation and positioning
   background = asset_registry.get_asset_by_name("kitchen")()
   pick_object = asset_registry.get_asset_by_name("cracker_box")()
   pick_object.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

   # Scene composition
   scene = Scene(assets=[background, pick_object])

   # Environment integration
   environment = IsaacArenaEnvironment(
       name="manipulation_task",
       embodiment=embodiment,
       scene=scene,  # Physical environment layout
       task=task,
       teleop_device=teleop_device
   )

Usage Examples
--------------

**Kitchen Pick and Place**

.. code-block:: python

   background = asset_registry.get_asset_by_name("kitchen")()
   mustard_bottle = asset_registry.get_asset_by_name("mustard_bottle")()
   mustard_bottle.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1)))

   scene = Scene(assets=[background, mustard_bottle])

**Microwave Interaction**

.. code-block:: python

   kitchen = asset_registry.get_asset_by_name("kitchen")()
   microwave = asset_registry.get_asset_by_name("microwave")()
   microwave.set_initial_pose(Pose(position_xyz=(0.8, 0.0, 0.23)))

   scene = Scene(assets=[kitchen, microwave])

**Object References**

.. code-block:: python

   # Reference elements within larger scene assets
   destination = ObjectReference(
       name="kitchen_drawer",
       prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
       parent_asset=kitchen,
       object_type=ObjectType.RIGID
   )
