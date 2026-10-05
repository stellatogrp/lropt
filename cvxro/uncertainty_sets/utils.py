def check_affine_transform(affine_transform):
    assert "b" in affine_transform
    assert "A" in affine_transform


def check_indices_dict(indices_dict, n_data):
    """
    Validate and normalize an indices_dict mapping keys 'train','test','validate'
    to numpy integer index arrays. Returns a dict with numpy arrays.

    Raises ValueError on out-of-bounds indices or overlapping indices.
    """
    import numpy as _np

    if indices_dict is None:
        return None

    if n_data is None:
        raise ValueError("Cannot validate indices_dict without " \
        "data (data is None)")

    provided = {}
    for key in ("train", "test", "validate"):
        if key in indices_dict and indices_dict[key] is not None:
            arr = _np.asarray(indices_dict[key])
            # boolean mask -> convert to integer indices
            if arr.dtype == bool:
                if arr.size != n_data:
                    raise ValueError(
                        f"Boolean mask for '{key}' has incorrect length "
                        f"{arr.size}, expected {n_data}"
                    )
                arr = _np.nonzero(arr)[0]
            else:
                arr = arr.astype(int)
            if arr.size > 0 and (_np.any(arr < 0) or _np.any(arr >= n_data)):
                raise ValueError(f"indices_dict['{key}'] contains out-of-bounds indices")
            provided[key] = arr

    # check overlaps
    if provided:
        all_provided = _np.concatenate([provided[k] for k in provided])
        if all_provided.size != 0 and _np.unique(all_provided).size != all_provided.size:
            raise ValueError("Overlapping indices found in indices_dict")

    return provided
