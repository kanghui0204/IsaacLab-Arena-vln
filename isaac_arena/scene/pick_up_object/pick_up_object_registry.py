import isaac_arena.scene.pick_up_object.pick_up_object as pick_up_objects

PICKUP_REGISTRY = {
    "mug": pick_up_objects.Mug,
    "gelatin_box": pick_up_objects.GelatinBox,
    "mac_and_cheese_box": pick_up_objects.MacandCheeseBox,
    "sugar_box": pick_up_objects.SugarBox,
    "tomato_soup_can": pick_up_objects.TomatoSoupCan,
    # add more pickup-objects here…
}


def get_pickup_object(name: str, **kwargs) -> pick_up_objects.PickUpObjects:
    try:
        cls = PICKUP_REGISTRY[name]
    except KeyError:
        raise ValueError(f"No pick-up object named {name!r}")
    return cls(**kwargs)
