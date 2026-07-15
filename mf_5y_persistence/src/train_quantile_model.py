class NotReadyError(Exception):
    pass

def train_quantile_model(*args, **kwargs):
    raise NotReadyError("train_quantile_model is not yet implemented pending training gate pass.")
