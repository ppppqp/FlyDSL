# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import importlib
import pickle
import threading
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Union

from .._mlir import ir
from .._mlir.execution_engine import ExecutionEngine

_GPU_MODULE_INIT = "flydsl_gpu_module_init"
_GPU_MODULE_LOAD_TO_DEVICE = "flydsl_gpu_module_load_to_device"


@dataclass(frozen=True)
class DeviceObject:
    data: bytes
    format: int
    target: str


@dataclass(frozen=True)
class LaunchArgument:
    """One physical device-kernel argument in a producer-side launch plan."""

    logical_index: int
    kind: str
    source_type: str
    binding: Optional[str] = None
    value: Optional[Union[int, float]] = None


@dataclass(frozen=True)
class KernelLaunch:
    """Restricted, runtime-independent description of one GPU launch."""

    id: int
    kernel: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_memory: int
    arguments: tuple[LaunchArgument, ...]
    dependencies: tuple[int, ...]


@dataclass(frozen=True)
class LaunchPlan:
    """A verified straight-line launch region exported by FlyDSL."""

    host_entry: str
    launches: tuple[KernelLaunch, ...]


@dataclass(frozen=True)
class OrchestrationArtifact:
    """Stable, runtime-independent output for an external orchestration layer.

    ``compiled_ir`` contains the final ``gpu.binary`` objects. ``source_ir`` retains
    the host-side ``gpu.launch_func`` structure. ``kernel_abi`` describes device
    entry points and must not be confused with the JIT host wrapper's C ABI.
    """

    compiled_ir: str
    source_ir: Optional[str]
    host_entry: str
    backend: str
    target: str
    kernel_abi: str
    device_objects: tuple[DeviceObject, ...]
    launch_plan: Optional[LaunchPlan] = None
    launch_plan_error: Optional[str] = None


class LaunchPlanError(ValueError):
    """Raised when source IR cannot be represented by the restricted plan ABI."""


def _operation_name(operation) -> str:
    generic_operation = getattr(operation, "operation", operation)
    return generic_operation.name


def _constant_number(value, description: str) -> Union[int, float]:
    owner = value.owner
    if not hasattr(owner, "name") or _operation_name(owner) != "arith.constant":
        raise LaunchPlanError(f"{description} must be an arith.constant")
    attribute = owner.attributes["value"]
    if isinstance(attribute, ir.IntegerAttr):
        return int(attribute)
    if isinstance(attribute, ir.FloatAttr):
        return float(attribute.value)
    raise LaunchPlanError(f"{description} has unsupported constant type {value.type}")


def _constant_int(value, description: str) -> int:
    result = _constant_number(value, description)
    if not isinstance(result, int):
        raise LaunchPlanError(f"{description} must be an integer")
    return result


def _walk_operations(operation):
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child
                yield from _walk_operations(child)


def _extract_launch_argument(value, logical_index: int, entry_block, bindings) -> LaunchArgument:
    source_type = str(value.type)
    owner = value.owner
    if owner == entry_block:
        name = value.get_name()
        try:
            binding = bindings[name]
        except KeyError as exc:
            raise LaunchPlanError(f"kernel argument {logical_index} is not a named host argument") from exc
        if source_type.startswith("!fly.ptr<"):
            if "global" not in source_type:
                raise LaunchPlanError(f"kernel argument {logical_index} uses non-global pointer type {source_type}")
            return LaunchArgument(logical_index, "resource", source_type, binding=binding)
        if source_type.startswith("!fly.memref<") or source_type.startswith("memref<"):
            raise LaunchPlanError(
                f"kernel argument {logical_index} uses {source_type}; memref ABI expansion is not yet supported"
            )
        if source_type in {"i32", "i64", "f32", "f64"}:
            return LaunchArgument(logical_index, "scalar", source_type, binding=binding)
        raise LaunchPlanError(f"kernel argument {logical_index} has unsupported type {source_type}")

    if hasattr(owner, "name") and _operation_name(owner) == "arith.constant":
        if source_type not in {"i32", "i64", "f32", "f64"}:
            raise LaunchPlanError(f"kernel argument {logical_index} has unsupported constant type {source_type}")
        return LaunchArgument(
            logical_index,
            "constant",
            source_type,
            value=_constant_number(value, f"kernel argument {logical_index}"),
        )
    raise LaunchPlanError(
        f"kernel argument {logical_index} is computed or view-derived; only direct host arguments and constants "
        "are supported"
    )


def extract_launch_plan(module, host_entry: str, binding_names) -> LaunchPlan:
    """Extract a conservative straight-line plan while FlyDSL owns the MLIR module.

    The source-order dependency chain deliberately preserves sequential host-launch
    semantics. Effect-derived dependency removal belongs in CKL after producer
    interfaces expose sufficient memory effects.
    """
    host_operation = None
    for operation in module.body.operations:
        if _operation_name(operation) != "func.func":
            continue
        symbol = ir.StringAttr(operation.attributes["sym_name"]).value
        if symbol == host_entry:
            host_operation = operation
            break
    if host_operation is None:
        raise LaunchPlanError(f"host entry @{host_entry} was not found")
    if len(host_operation.regions[0].blocks) != 1:
        raise LaunchPlanError("host entry must have one block")

    entry_block = host_operation.regions[0].blocks[0]
    if len(entry_block.arguments) != len(binding_names):
        raise LaunchPlanError("host argument metadata does not match the entry block")
    bindings = {argument.get_name(): name for argument, name in zip(entry_block.arguments, binding_names)}

    direct_launches = [
        operation for operation in entry_block.operations if _operation_name(operation) == "gpu.launch_func"
    ]
    all_launches = [
        operation for operation in _walk_operations(host_operation) if _operation_name(operation) == "gpu.launch_func"
    ]
    if len(direct_launches) != len(all_launches):
        raise LaunchPlanError("gpu.launch_func nested in control flow is not supported")
    if not direct_launches:
        raise LaunchPlanError("host entry contains no gpu.launch_func operations")

    launches = []
    for launch_id, operation in enumerate(direct_launches):
        launch = getattr(operation, "opview", operation)
        if launch.clusterSizeX is not None or launch.cooperative:
            raise LaunchPlanError("clustered and cooperative launches are not supported")
        grid = tuple(
            _constant_int(value, f"launch {launch_id} grid dimension")
            for value in (launch.gridSizeX, launch.gridSizeY, launch.gridSizeZ)
        )
        block = tuple(
            _constant_int(value, f"launch {launch_id} block dimension")
            for value in (launch.blockSizeX, launch.blockSizeY, launch.blockSizeZ)
        )
        if any(value <= 0 for value in (*grid, *block)):
            raise LaunchPlanError(f"launch {launch_id} grid and block dimensions must be positive")
        shared_memory = (
            0
            if launch.dynamicSharedMemorySize is None
            else _constant_int(launch.dynamicSharedMemorySize, f"launch {launch_id} shared memory")
        )
        if shared_memory < 0:
            raise LaunchPlanError(f"launch {launch_id} shared memory must be non-negative")
        arguments = tuple(
            _extract_launch_argument(value, index, entry_block, bindings)
            for index, value in enumerate(launch.kernelOperands)
        )
        launches.append(
            KernelLaunch(
                id=launch_id,
                kernel=str(launch.kernel),
                grid=grid,
                block=block,
                shared_memory=shared_memory,
                arguments=arguments,
                dependencies=() if launch_id == 0 else (launch_id - 1,),
            )
        )

    supported_host_operations = {"arith.constant", "gpu.launch_func", "func.return"}
    for operation in entry_block.operations:
        name = _operation_name(operation)
        if name not in supported_host_operations:
            raise LaunchPlanError(f"host operation {name} is not supported in an exported launch plan")
    return LaunchPlan(host_entry=host_entry, launches=tuple(launches))


def _extract_device_objects(compiled_module) -> tuple[DeviceObject, ...]:
    """Copy gpu.binary payloads while the producing MLIR runtime owns the module."""
    operation = getattr(compiled_module, "operation", None)
    if operation is None:
        return ()

    from .._mlir.dialects import gpu

    objects = []

    def visit(op):
        if op.name == "gpu.binary":
            for attribute in ir.ArrayAttr(op.attributes["objects"]):
                gpu_object = gpu.ObjectAttr(attribute)
                objects.append(
                    DeviceObject(
                        data=gpu_object.object,
                        format=int(gpu_object.format),
                        target=str(gpu_object.target),
                    )
                )
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child)

    visit(operation)
    return tuple(objects)


def _qualname(fn: Callable) -> Optional[str]:
    """Serialise a callable as ``module:qualname``; return None if not possible.

    Rejects anything that can't round-trip through ``_resolve_qualname``:

    - Missing ``__module__`` / ``__qualname__``.
    - ``__qualname__`` containing ``<`` (lambdas → ``<lambda>``,
      nested / closure functions → ``<locals>``, comprehensions, etc.).
    - Bound methods: ``__qualname__`` looks fine (``Class.method``) but
      resolving it yields the *unbound* function, silently dropping ``self``.

    As a final safety net we verify the generated ref actually resolves
    back to *fn* in the current process — if not, we refuse.
    """
    mod = getattr(fn, "__module__", None)
    qn = getattr(fn, "__qualname__", None)
    if not mod or not qn:
        return None
    if "<" in qn:
        return None
    if getattr(fn, "__self__", None) is not None:
        return None
    ref = f"{mod}:{qn}"
    # Round-trip check: guarantees the symbol is actually reachable
    # under the name we plan to write into the pickle stream.
    if _resolve_qualname(ref) is not fn:
        return None
    return ref


def _resolve_qualname(ref: str) -> Optional[Callable]:
    """Inverse of _qualname; silently returns None on failure."""
    try:
        mod_name, qn = ref.split(":", 1)
        obj = importlib.import_module(mod_name)
        for part in qn.split("."):
            obj = getattr(obj, part)
        return obj
    except Exception:
        return None


@lru_cache(maxsize=1)
def _resolve_runtime_libs() -> List[str]:
    from .backends import get_backend

    backend = get_backend()
    mlir_libs_dir = Path(__file__).resolve().parent.parent / "_mlir" / "_mlir_libs"
    libs = [mlir_libs_dir / name for name in backend.jit_runtime_lib_basenames()]
    for lib in libs:
        if not lib.exists():
            raise FileNotFoundError(f"Required JIT runtime library not found: {lib}\nPlease rebuild the project.")
    return [str(p) for p in libs]


def _pack_ciface_args(*args) -> ctypes.c_void_p:
    """Pack args for MLIR's packed-args wrapper (one holder cell per arg)."""
    holders = [ctypes.c_void_p(ctypes.addressof(a)) for a in args]
    packed = (ctypes.c_void_p * len(args))()
    for i, h in enumerate(holders):
        packed[i] = ctypes.addressof(h)
    result = ctypes.c_void_p(ctypes.addressof(packed))
    result._keepalive = (packed, holders, args)  # keep intermediates alive  # type: ignore[attr-defined]
    return result


class GpuJitModule:
    """Owns GPU modules loaded for one ExecutionEngine."""

    def __init__(self, engine: ExecutionEngine, modules: List[int]):
        self.engine = engine
        self.modules = list(modules)
        self._runtime_lib = ctypes.CDLL(str(_resolve_runtime_libs()[0]))
        self._runtime_lib.mgpuModuleUnload.argtypes = [ctypes.c_void_p]
        self._runtime_lib.mgpuModuleUnload.restype = None
        self._unloaded = False

    def unload(self) -> None:
        if self._unloaded:
            return
        try:
            for module in self.modules:
                if module:
                    self._runtime_lib.mgpuModuleUnload(ctypes.c_void_p(module))
            self.modules.clear()
        finally:
            self._unloaded = True

    def __del__(self):
        self.unload()


def _load_gpu_modules(engine: ExecutionEngine) -> List[int]:
    """Load embedded GPU modules through explicit symbols emitted by FlyDSL."""
    init_ptr = engine.raw_lookup(_GPU_MODULE_INIT)
    load_ptr = engine.raw_lookup(_GPU_MODULE_LOAD_TO_DEVICE)
    if not init_ptr or not load_ptr:
        raise RuntimeError(
            "compiled module does not expose FlyDSL ROCm module loader symbols; "
            "make sure the FlyDSL ROCm explicit-module offloading handler is registered"
        )

    init = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(init_ptr)
    load = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(load_ptr)

    module = ctypes.c_void_p()
    err = ctypes.c_int32(0)
    packed_ciface_arg = _pack_ciface_args(module, err)
    init(packed_ciface_arg)
    if err.value != 0:
        raise RuntimeError(f"{_GPU_MODULE_INIT} failed with error code {err.value}")
    load(packed_ciface_arg)
    if err.value != 0 or not module.value:
        raise RuntimeError(f"{_GPU_MODULE_LOAD_TO_DEVICE} failed with error code {err.value}")
    return [module.value]


def build_abi_storage(ctypes_seq):
    """One zeroed ctypes storage per slot ctype, plus a packed pointer array of their addresses."""
    packed = (ctypes.c_void_p * len(ctypes_seq))()
    storages = []
    for i, ct in enumerate(ctypes_seq):
        try:
            s = ct(0)
        except TypeError:
            s = ct()
        storages.append(s)
        packed[i] = ctypes.addressof(s)
    return storages, packed


def _build_dispatch_factory(slot_specs):
    """Generate a straight-line dispatch-closure factory for ``slot_specs``.

    Returns ``make(packed, storages, func_exe) -> dispatch(args_tuple)``.  The
    generated ``dispatch`` unrolls every slot -- no per-slot Python loop, branch,
    or tuple-unpack -- with per-slot storages and fill fns bound as closure locals.
    """
    setup, body, fills = [], [], []
    for i, (arg_idx, _, fill) in enumerate(slot_specs):
        if fill is None:
            continue  # null slot (auto-stream): packed[i] stays NULL after alloc
        fi = len(fills)
        fills.append(fill)
        setup.append(f"    s{i} = storages[{i}]")
        setup.append(f"    f{i} = fills[{fi}]")
        body.append(f"        f{i}(a[{arg_idx}], s{i})")

    src = "def make(packed, storages, func_exe, fills):\n"
    src += "".join(line + "\n" for line in setup)
    src += "    def dispatch(a):\n"
    src += "".join(line + "\n" for line in body)
    src += "        return func_exe(packed)\n"
    src += "    return dispatch\n"

    ns = {}
    exec(compile(src, "<flydsl-dispatch>", "exec"), ns)
    make = ns["make"]
    fills = tuple(fills)

    def factory(packed, storages, func_exe):
        return make(packed, storages, func_exe, fills)

    return factory


class CallState:
    """Pre-allocated state for fast kernel dispatch -- the single storage + fill
    dispatch implementation.

    each call then runs only the unrolled per-slot fills and invokes the JIT'd function
    -- no per-slot loop, no ctypes allocation. Thread-local for thread safety.
    """

    __slots__ = ("_func_exe", "_spec", "_tls", "_factory")

    def __init__(self, slot_specs, func_exe):
        self._func_exe = func_exe
        self._spec = slot_specs  # list of (arg_idx, ctype, fill)
        self._tls = threading.local()
        self._factory = _build_dispatch_factory(slot_specs)

    def _make_dispatch(self):
        # Allocate one typed storage per slot + the packed pointer array; the null
        # auto-stream slot uses c_void_p -> NULL (its fill is None, never written).
        storages, packed = build_abi_storage([ctype for _arg_idx, ctype, _fill in self._spec])
        # The dispatch closure keeps packed + storages alive
        self._tls.packed = packed
        self._tls.storages = storages
        return self._factory(packed, storages, self._func_exe)

    def __call__(self, args_tuple):
        dispatch = getattr(self._tls, "dispatch", None)
        if dispatch is None:
            dispatch = self._tls.dispatch = self._make_dispatch()
        return dispatch(args_tuple)


class CompiledArtifact:
    def __init__(
        self,
        compiled_module: ir.Module,
        func_name: str,
        source_ir: Optional[str] = None,
        post_load_processors: Optional[List[Callable]] = None,
        link_libs: Optional[List[str]] = None,
        uses_explicit_module: bool = False,
        backend: str = "",
        target: str = "",
        kernel_abi: str = "",
        launch_plan: Optional[LaunchPlan] = None,
        launch_plan_error: Optional[str] = None,
    ):
        self._ir_text = str(compiled_module)
        self._device_objects = _extract_device_objects(compiled_module)
        self._entry = func_name
        self._source_ir = source_ir
        self._post_load_processors = post_load_processors or []
        self._link_libs = link_libs or []
        self._uses_explicit_module = uses_explicit_module
        self._backend = backend
        self._target = target
        self._kernel_abi = kernel_abi
        self._launch_plan = launch_plan
        self._launch_plan_error = launch_plan_error
        self._module = None
        self._engine = None
        self._jit_module = None
        self._func_exe = None
        self._lock = threading.Lock()

    def __getstate__(self):
        # Both the compiled MLIR and source IR compress well. The source IR is
        # retained because external orchestration exports need its gpu.launch_func
        # structure after a disk-cache hit.
        #
        # Serialise post-load processors by fully-qualified name so the pickle
        # stream carries no concrete callables.
        #
        # If any processor cannot be represented as module:qualname
        # (e.g. lambdas, functools.partial, bound methods) we *refuse*
        # to pickle instead of silently dropping it.  Silently dropping
        # would let the disk cache round-trip a kernel that later
        # launches without running its required initialiser, and the
        # first kernel launch on the next process would GPU-fault on
        # uninitialised device-side globals — with a stack that gives
        # no hint about the missing processor.  Raising here means:
        # (a) the failure is visible during cache-write, before any
        # user sees a mysterious fault, and (b) the caller (typically
        # the on-disk cache layer) can fall back to "memory cache
        # only" for this artifact without crashing the run.
        refs: List[str] = []
        unpicklable: List[str] = []
        for p in self._post_load_processors:
            ref = _qualname(p)
            if ref is None:
                unpicklable.append(repr(p))
            else:
                refs.append(ref)
        if unpicklable:
            raise pickle.PicklingError(
                "CompiledArtifact has post-load processors that are not "
                "representable as module:qualname: "
                f"{unpicklable}.  Wrap them in a top-level function in a "
                "regular Python module so the on-disk cache can re-import "
                "them after pickle round-trip (lambdas, partial, and bound "
                "methods are not supported).  If this artifact genuinely "
                "must not be cached to disk, suppress the disk-cache write "
                "path for it."
            )
        state = {
            "ir_zlib": zlib.compress(self._ir_text.encode("utf-8")),
            "entry": self._entry,
            "processor_refs": refs,
            "link_libs": self._link_libs,
            "uses_explicit_module": self._uses_explicit_module,
            "backend": self._backend,
            "target": self._target,
            "kernel_abi": self._kernel_abi,
            "device_objects": self._device_objects,
            "launch_plan": self._launch_plan,
            "launch_plan_error": self._launch_plan_error,
        }
        if self._source_ir is not None:
            state["source_ir_zlib"] = zlib.compress(self._source_ir.encode("utf-8"))
        return state

    def __setstate__(self, state):
        # Accept the uncompressed format written by older FlyDSL versions.
        if "ir_zlib" in state:
            self._ir_text = zlib.decompress(state["ir_zlib"]).decode("utf-8")
        else:
            self._ir_text = state["ir_text"]
        self._entry = state["entry"]
        if "source_ir_zlib" in state:
            self._source_ir = zlib.decompress(state["source_ir_zlib"]).decode("utf-8")
        else:
            self._source_ir = state.get("source_ir")
        self._link_libs = state.get("link_libs", [])
        self._uses_explicit_module = state.get("uses_explicit_module", False)
        self._backend = state.get("backend", "")
        self._target = state.get("target", "")
        self._kernel_abi = state.get("kernel_abi", "")
        self._device_objects = tuple(state.get("device_objects", ()))
        self._launch_plan = state.get("launch_plan")
        self._launch_plan_error = state.get("launch_plan_error")
        self._post_load_processors = []
        missing: List[str] = []
        for ref in state.get("processor_refs", []):
            fn = _resolve_qualname(ref)
            if fn is None:
                missing.append(ref)
            else:
                self._post_load_processors.append(fn)
        if missing:
            raise pickle.UnpicklingError(
                "CompiledArtifact could not resolve required post-load " f"processors: {missing}"
            )
        self._module = None
        self._engine = None
        self._jit_module = None
        self._func_exe = None
        self._lock = threading.Lock()

    def _ensure_engine(self):
        with self._lock:
            if self._engine is not None:
                return

            # Keep the Context alive on self._ctx: destroying it
            # while ExecutionEngine still holds HSA code objects
            # causes GPU memory access faults.
            from .jit_function import _create_mlir_context

            ctx = _create_mlir_context()
            with ctx:
                module = ir.Module.parse(self._ir_text)
                engine = ExecutionEngine(
                    module,
                    opt_level=3,
                    shared_libs=_resolve_runtime_libs(),
                )
                engine.initialize()

            if self._uses_explicit_module:
                loaded_modules = _load_gpu_modules(engine)

                # Post-condition: if callers registered post-load
                # processors, at least one module load MUST have been
                # observed.  Zero observations means the explicit module
                # loader did not run; fail loud here rather than let
                # kernel launch segfault on uninitialised device globals.
                if self._post_load_processors and not loaded_modules:
                    raise RuntimeError(
                        "post_load_processors registered but no hipModuleLoad "
                        "was observed during ExecutionEngine.initialize(). "
                        "Device-side globals (e.g. mori shmem's "
                        "globalGpuStates) will be uninitialised at kernel "
                        "launch.  Check that the compiled module contains a GPU "
                        "binary and that mgpuModuleLoad is still the ROCm runtime "
                        "loader symbol used by MLIR."
                    )

                jit_module = GpuJitModule(engine, loaded_modules)
                for proc in self._post_load_processors:
                    for mod_handle in loaded_modules:
                        proc(mod_handle)
            else:
                jit_module = None

            self._module = module
            self._engine = engine
            self._jit_module = jit_module
            self._ctx = ctx

    def _get_func_exe(self):
        if self._func_exe is None:
            if self._engine is None:
                self._ensure_engine()
            func_ptr = self._engine.raw_lookup(self._entry)
            self._func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)
        return self._func_exe

    def dump(self, compiled: bool = True):
        if compiled:
            print("=" * 60)
            print("Compiled MLIR IR:")
            print("=" * 60)
            print(self._ir_text)
        else:
            if self._source_ir is None:
                print("Original IR not available")
            else:
                print("=" * 60)
                print("Original MLIR IR:")
                print("=" * 60)
                print(self._source_ir)

    @property
    def ir(self) -> str:
        return self._ir_text

    @property
    def source_ir(self) -> Optional[str]:
        return self._source_ir

    @property
    def entry(self) -> str:
        return self._entry

    def export_for_orchestration(self) -> OrchestrationArtifact:
        """Return public compiler output without initializing FlyDSL's runtime."""
        if not self._backend or not self._target or not self._kernel_abi:
            raise RuntimeError(
                "compiled artifact predates orchestration metadata; recompile it with the current FlyDSL"
            )
        return OrchestrationArtifact(
            compiled_ir=self._ir_text,
            source_ir=self._source_ir,
            host_entry=self._entry,
            backend=self._backend,
            target=self._target,
            kernel_abi=self._kernel_abi,
            device_objects=self._device_objects,
            launch_plan=self._launch_plan,
            launch_plan_error=self._launch_plan_error,
        )
