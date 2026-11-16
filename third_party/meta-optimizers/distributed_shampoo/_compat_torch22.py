"""torch<2.5 compatibility shims (LOCAL ADDITION, not Meta's — see VENDORED.txt).

This repo runs on torch 2.2.2 on Intel macOS (the last x86_64 build) and on
current torch on Lambda. Upstream distributed_shampoo targets torch>=2.8; the
only APIs it needs that 2.2.2 lacks are:

  1. The public DTensor namespace. torch 2.2 ships every needed name
     (DTensor, zeros, distribute_tensor, placements, ...) under
     `torch.distributed._tensor`; `torch.distributed.tensor` exists but holds
     only the `parallel` subpackage. We copy `_tensor`'s public names INTO
     the existing `torch.distributed.tensor` module (never replace it —
     torch's own fsdp internals import `torch.distributed.tensor.parallel`
     through its real __path__) and alias the one submodule upstream imports
     (`placement_types`) in sys.modules.

  2. FSDP2 (`fully_shard` / `FSDPModule`), which does not exist in 2.2 in any
     namespace. We inject inert stubs: `FSDPModule` is an empty class — an
     isinstance() check against it is always False, which is exactly right on
     a torch that cannot have FSDP2 modules — and `fully_shard` raises if
     actually called. This keeps upstream's eager imports working while the
     FSDP2 code paths stay unreachable.

On torch >= 2.5 both shims detect the real APIs and do nothing.
"""

from __future__ import annotations

import sys
import types

import torch.distributed.tensor as _new_tensor_ns

if not hasattr(_new_tensor_ns, "DTensor"):  # torch < 2.5
    import torch.distributed._tensor as _old_tensor_ns
    import torch.distributed._tensor.placement_types as _old_placement_types

    for _name in dir(_old_tensor_ns):
        if not _name.startswith("_") and not hasattr(_new_tensor_ns, _name):
            setattr(_new_tensor_ns, _name, getattr(_old_tensor_ns, _name))
    # torch 2.2's zeros/empty/full live in _tensor's namespace via its api
    # module; make sure the factory functions upstream uses are present.
    for _name in ("zeros", "empty", "full", "ones", "rand", "randn"):
        if hasattr(_old_tensor_ns, _name) and not hasattr(_new_tensor_ns, _name):
            setattr(_new_tensor_ns, _name, getattr(_old_tensor_ns, _name))
    sys.modules.setdefault(
        "torch.distributed.tensor.placement_types", _old_placement_types
    )
    _new_tensor_ns.placement_types = _old_placement_types  # type: ignore[attr-defined]

try:
    from torch.distributed.fsdp import FSDPModule as _FSDPModule  # noqa: F401
except ImportError:  # torch < 2.6: no FSDP2
    import torch.distributed._composable as _composable
    import torch.distributed.fsdp as _fsdp

    class _StubFSDPModule:
        """Stands in for FSDP2's FSDPModule; no instance can ever exist."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("FSDP2 (FSDPModule) requires torch >= 2.6")

    def _stub_fully_shard(*args: object, **kwargs: object) -> None:
        raise RuntimeError("FSDP2 (fully_shard) requires torch >= 2.6")

    _fsdp.FSDPModule = _StubFSDPModule  # type: ignore[attr-defined]
    _fsdp.fully_shard = _stub_fully_shard  # type: ignore[attr-defined]

    _composable_fsdp = types.ModuleType("torch.distributed._composable.fsdp")
    _composable_fsdp.fully_shard = _stub_fully_shard  # type: ignore[attr-defined]
    _composable_fsdp.FSDPModule = _StubFSDPModule  # type: ignore[attr-defined]
    sys.modules["torch.distributed._composable.fsdp"] = _composable_fsdp
    _composable.fsdp = _composable_fsdp  # type: ignore[attr-defined]
