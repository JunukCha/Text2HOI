import numpy as np

def process_text(
    action_name, obj_name, 
    is_lhand, is_rhand, 
    text_descriptions, return_key=False, 
):
    if is_lhand and is_rhand:
        text = f"{action_name} {obj_name} with both hands."
    elif is_lhand:
        text = f"{action_name} {obj_name} with left hand."
    elif is_rhand:
        text = f"{action_name} {obj_name} with right hand."
    text_key = text.capitalize()
    if return_key:
        return text_key
    else:
        text_description = text_descriptions[text_key]
        text = np.random.choice(text_description)
        return text