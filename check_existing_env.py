
import os
import sys
import subprocess

print("=" * 80)
print("BASIC ENV")
print("=" * 80)

print("python:", sys.executable)
print("python version:", sys.version)
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
print("CUDA_MODULE_LOADING:", os.environ.get("CUDA_MODULE_LOADING"))

print("\n" + "=" * 80)
print("NVIDIA-SMI")
print("=" * 80)

try:
    out = subprocess.check_output(["nvidia-smi"], text=True)
    print(out)
except Exception as e:
    print("nvidia-smi failed:", repr(e))

print("\n" + "=" * 80)
print("PYTHON PACKAGES")
print("=" * 80)

packages = [
    "torch",
    "torchvision",
    "mmcv",
    "mmengine",
    "timm",
    "transformers",
    "cv2",
    "scipy",
    "numpy",
    "pandas",
]

for p in packages:
    try:
        mod = __import__(p)
        print(f"{p}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        print(f"{p}: import failed -> {repr(e)}")

print("\n" + "=" * 80)
print("TORCH CUDA INFO")
print("=" * 80)

try:
    import torch
    import torchvision

    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch cuda available:", torch.cuda.is_available())
    print("torch cudnn version:", torch.backends.cudnn.version())
    print("torch cudnn enabled:", torch.backends.cudnn.enabled)
    print("torch cudnn benchmark:", torch.backends.cudnn.benchmark)
    print("torch cudnn deterministic:", torch.backends.cudnn.deterministic)
    print("torchvision:", torchvision.__version__)

    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        print("device 0:", torch.cuda.get_device_name(0))
        print("device capability:", torch.cuda.get_device_capability(0))
        print("current device:", torch.cuda.current_device())
        print("memory allocated:", torch.cuda.memory_allocated(0))
        print("memory reserved:", torch.cuda.memory_reserved(0))

except Exception as e:
    print("torch cuda info failed:", repr(e))

print("\n" + "=" * 80)
print("CUDA BASIC TEST: cuDNN ON")
print("=" * 80)

try:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.empty_cache()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    x = torch.randn(2, 3, 352, 352, device="cuda")
    conv = torch.nn.Conv2d(3, 32, 3, padding=1).cuda()

    y = conv(x)
    loss = y.mean()
    loss.backward()

    torch.cuda.synchronize()

    print("[OK] cuDNN ON conv backward passed:", tuple(y.shape))

except Exception as e:
    print("[FAIL] cuDNN ON conv backward failed:", repr(e))

print("\n" + "=" * 80)
print("CUDA BASIC TEST: cuDNN OFF")
print("=" * 80)

try:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.empty_cache()

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    x = torch.randn(2, 3, 352, 352, device="cuda")
    conv = torch.nn.Conv2d(3, 32, 3, padding=1).cuda()

    y = conv(x)
    loss = y.mean()
    loss.backward()

    torch.cuda.synchronize()

    print("[OK] cuDNN OFF conv backward passed:", tuple(y.shape))

except Exception as e:
    print("[FAIL] cuDNN OFF conv backward failed:", repr(e))

print("\n" + "=" * 80)
print("TORCHVISION DEFORMCONV TEST")
print("=" * 80)

try:
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.empty_cache()

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    x = torch.randn(2, 16, 64, 64, device="cuda", requires_grad=True)
    offset = torch.randn(2, 18, 64, 64, device="cuda")
    layer = torchvision.ops.DeformConv2d(
        16,
        16,
        kernel_size=3,
        padding=1,
    ).cuda()

    y = layer(x, offset)
    loss = y.mean()
    loss.backward()

    torch.cuda.synchronize()

    print("[OK] torchvision DeformConv2d passed:", tuple(y.shape))

except Exception as e:
    print("[FAIL] torchvision DeformConv2d failed:", repr(e))

print("\n" + "=" * 80)
print("CUDA MEMORY SUMMARY")
print("=" * 80)

try:
    import torch

    if torch.cuda.is_available():
        print(torch.cuda.memory_summary(device=0, abbreviated=True))
    else:
        print("CUDA not available")

except Exception as e:
    print("memory summary failed:", repr(e))

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
