import datetime
import logging

import torch
import torch.nn as nn


def str2bool(s):
    return s.lower() in ('true', '1')


def _module_params(module):
    return sum(p.numel() for p in module.parameters())


def _conv_signature(module):
    """Describe a block by its convolutions: [in_ch, out_ch, stride]."""
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        return "-"
    first, last = convs[0], convs[-1]
    stride = max(max(c.stride) if isinstance(c.stride, tuple) else c.stride for c in convs)
    return f"[{first.in_channels}, {last.out_channels}, s{stride}]"


def _block_name(module):
    cls = module.__class__.__name__
    if isinstance(module, nn.Sequential):
        n_children = len(module)
        if n_children == 3:
            return "conv_bn (Conv+BN+ReLU)"
        if n_children == 6:
            return "conv_dw (depthwise separable)"
        return f"Sequential[{n_children}]"
    return cls


def profile_gflops(net, input_size=(3, 640, 640), device="cpu"):
    """Estimate forward GFLOPs (2 x MACs, conv + linear) with forward hooks.

    Dependency-free replacement for thop/ptflops, same convention as
    ultralytics (FLOPs = 2 * multiply-accumulates), batch size 1.
    """
    macs = [0]
    hooks = []

    def hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            macs[0] += out.numel() * (module.in_channels // module.groups) * kh * kw
        elif isinstance(module, nn.Linear):
            macs[0] += out.numel() * module.in_features

    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    was_training = net.training
    net.eval()
    with torch.no_grad():
        net(torch.zeros(1, *input_size, device=device))
    for h in hooks:
        h.remove()
    net.train(was_training)
    return 2 * macs[0] / 1e9


def print_model_summary(net, input_size=(3, 640, 640), device="cpu",
                        model_name="RFB-640-landmark"):
    """Print a YOLO-style per-block table and summary line for the SSD model.

    Expects the SSD from vision/ssd/ssd.py (base_net + extras + header
    ModuleLists). Columns mirror ultralytics: index, from, n (module count in
    the row), params, module, arguments = [in_ch, out_ch, max stride].
    """
    rows = []
    for i, m in enumerate(net.base_net):
        rows.append((i, -1, 1, _module_params(m), _block_name(m), _conv_signature(m)))
    extras_idx = len(net.base_net)
    rows.append((extras_idx, extras_idx - 1, 1, _module_params(net.extras),
                 "extras (Conv1x1+SepConv)", _conv_signature(net.extras)))

    # feature-map tap points: outputs taken after base_net[idx-1] + extras
    taps = [idx - 1 for idx in net.source_layer_indexes if isinstance(idx, int)] + [extras_idx]
    header_specs = [
        ("classification_headers", net.classification_headers, "-> priors*2 (cls)"),
        ("regression_headers", net.regression_headers, "-> priors*4 (box)"),
        ("landmark_headers (NEW)", getattr(net, "landmark_headers", None), "-> priors*10 (5 kp)"),
    ]
    idx = extras_idx + 1
    for name, module_list, out_desc in header_specs:
        if module_list is None:
            continue
        rows.append((idx, str(taps), len(module_list), _module_params(module_list),
                     name, out_desc))
        idx += 1

    print(f"{'':>19}from  n    params  module{'':<34}arguments")
    for i, frm, n, params, name, args in rows:
        print(f"{i:>3}{str(frm):>20}{n:>3}{params:>10}  {name:<40}{args}")

    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_layers = len(list(net.modules()))
    gflops = profile_gflops(net, input_size, device)
    print(f"{model_name} summary: {n_layers} layers, {total:,} parameters, "
          f"{trainable:,} gradients, {gflops:.2f} GFLOPs (input {input_size[0]}x{input_size[1]}x{input_size[2]})")


_DTYPE_ALIASES = {
    "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
}


def resolve_device_dtype(precision="fp32", use_cuda=True, device=None):
    """Pick a (device, dtype) pair from a precision string, with safe fallbacks.

    Supports full precision plus both half-precision formats:
        "fp32" / "float32"            -> torch.float32
        "fp16" / "float16" / "half"   -> torch.float16
        "bf16" / "bfloat16"           -> torch.bfloat16

    Fallback rules (a warning is logged whenever a fallback triggers):
        - fp16 on CPU -> fp32 (CPU has poor/absent fp16 kernels for conv).
        - bf16 on a GPU without bf16 support -> fp16.

    Args:
        precision: one of the aliases above (case-insensitive).
        use_cuda: prefer CUDA when available (ignored if `device` is given).
        device: optional explicit device ("cuda:0", "cpu", or torch.device);
            overrides `use_cuda`.
    Returns:
        (device, dtype): a torch.device and the resolved torch dtype.
    """
    key = str(precision).lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"Unknown precision '{precision}'. Use one of: fp32, fp16, bf16.")
    dtype = _DTYPE_ALIASES[key]

    if device is not None:
        device = torch.device(device)
    else:
        device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")

    if dtype is torch.float16 and device.type == "cpu":
        logging.warning("fp16 is poorly supported on CPU; falling back to fp32.")
        dtype = torch.float32
    if dtype is torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        logging.warning("This GPU does not support bf16; falling back to fp16.")
        dtype = torch.float16
    return device, dtype


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = datetime.datetime.now()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = datetime.datetime.now() - self.clock[key]
        del self.clock[key]
        return interval.total_seconds()
        

def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))
