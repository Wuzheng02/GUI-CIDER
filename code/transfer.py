import re

def transfer_qwen3vl2atlas(action):
    if action == "PRESS_HOME":
        return "PRESS_HOME"
    elif action == "PRESS_BACK":
        return "PRESS_BACK"
    elif action.startswith("TASK_COMPLETE"):
        return "COMPLETE"
    elif action.startswith("WAIT"):
        return "WAIT"
    elif action.startswith("SWIPE"):
        return action.replace("SWIPE", "SCROLL ")
    elif action.startswith("TYPE"):
        return action.replace("TYPE", "TYPE ", 1)
    else:
        match = re.match(r'(CLICK|LONG_PRESS)\[(\d+)\s*,\s*(\d+)\]', action)
        if match:
            action_type = match.group(1)
            x = int(match.group(2))
            y = int(match.group(3))
            return f"{action_type} <point>[[{x},{y}]]</point>"
    
    return action 