def _ensure_stateful(inputs):
    if isinstance(inputs, list):
        raise TypeError('_ensure_statefule does not accept list.')
    if not isinstance(inputs, tuple) or len(inputs) != 2:
        inputs = (inputs, None)
    return inputs