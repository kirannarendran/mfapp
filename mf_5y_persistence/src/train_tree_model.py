class NotReadyError(Exception):
    pass

def train_tree_model(*args, **kwargs):
    raise NotReadyError("train_tree_model is not yet implemented pending training gate pass.")
