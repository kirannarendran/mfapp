class NotReadyError(Exception):
    pass

def evaluate_model(*args, **kwargs):
    raise NotReadyError("evaluate_model is not yet implemented pending training gate pass.")
