import torch

_LIB_HOLDER = []
_PATCHED = False

def ensure_torchvision_compat():
    global _PATCHED
    try:
        lib = torch.library.Library('torchvision', 'DEF')
        lib.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor')
        _LIB_HOLDER.append(lib)
    except Exception:
        pass

    if not _PATCHED:
        orig_register_fake = torch.library.register_fake
        def safe_register_fake(op_name, *args, **kwargs):
            decorator = orig_register_fake(op_name, *args, **kwargs)
            def wrapped(fn):
                try:
                    return decorator(fn)
                except RuntimeError as e:
                    if 'torchvision::nms' in str(e):
                        return fn
                    raise
            return wrapped
        torch.library.register_fake = safe_register_fake
        _PATCHED = True
