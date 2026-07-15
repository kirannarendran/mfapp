class NotReadyError(Exception):
    pass

def train_hierarchical_model(*args, **kwargs):
    raise NotReadyError("train_hierarchical_model is not yet implemented pending training gate pass.")
