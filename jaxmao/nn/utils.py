def calculate_same_padding(input_shape, kernel_size, strides):
    pad_along_height = max((input_shape[0] - 1) * strides[0] + kernel_size[0] - input_shape[0], 0)
    pad_along_width  = max((input_shape[1] - 1) * strides[1] + kernel_size[1] - input_shape[1], 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return ((pad_top, pad_bottom), (pad_left, pad_right))