"""Build script for the Col-Bandit C extension (colbandit._kernel).

The kernel sources live under native/numkong/ — a vendored snapshot of upstream
NumKong (https://github.com/ashvardanian/NumKong, Apache-2.0) plus the
colbandit_flat entry point in native/numkong/python/colbandit.c. We compile
them as a private extension module shipped inside the colbandit package, so
end users only need `pip install colbandit` — no separate numkong dependency.

The architecture/ISA probe table and per-platform compile flags below are
adapted from native/numkong/setup.py with file paths rebased to NK_ROOT.
"""

from __future__ import annotations

import glob
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Where the vendored NumKong source tree lives (relative to this setup.py).
NK_ROOT = "native/numkong"


# ---------------------------------------------------------------------------
# macOS DEVELOPER_DIR sanity (lifted from upstream)
# ---------------------------------------------------------------------------
if sys.platform == "darwin":
    _bad_dev_dir = os.environ.get("DEVELOPER_DIR")
    if _bad_dev_dir and (_bad_dev_dir == "public" or not Path(_bad_dev_dir).exists()):
        print(f"[colbandit] Ignoring invalid DEVELOPER_DIR={_bad_dev_dir!r}")
        os.environ.pop("DEVELOPER_DIR", None)


# ---------------------------------------------------------------------------
# Architecture detection (with env-var overrides for cross-compilation)
# ---------------------------------------------------------------------------
def is_64bit_x86() -> bool:
    override = os.environ.get("NK_TARGET_X86_")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("x86_64", "x64", "amd64")) and (sys.maxsize > 2**32)


def is_64bit_arm() -> bool:
    override = os.environ.get("NK_TARGET_ARM_")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("arm64", "aarch64")) and (sys.maxsize > 2**32)


def is_64bit_riscv() -> bool:
    override = os.environ.get("NK_TARGET_RISCV_")
    if override is not None:
        return override == "1"
    return (platform.machine().lower() in ("riscv64",)) and (sys.maxsize > 2**32)


def is_64bit_loongarch() -> bool:
    override = os.environ.get("NK_TARGET_LOONGARCH_")
    if override is not None:
        return override == "1"
    return (platform.machine().lower() in ("loongarch64",)) and (sys.maxsize > 2**32)


def is_64bit_power() -> bool:
    override = os.environ.get("NK_TARGET_POWER_")
    if override is not None:
        return override == "1"
    return (platform.machine().lower() in ("ppc64le", "ppc64", "powerpc64le", "powerpc64")) and (sys.maxsize > 2**32)


def is_wasm() -> bool:
    host = os.environ.get("_PYTHON_HOST_PLATFORM", "")
    return "emscripten" in host or "wasm" in host


def detect_cc():
    if sys.platform == "win32":
        return ("cl.exe", True)
    cc = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
    return (cc.split()[0], False)


def probe_isa(cc, probe_file, flags, is_msvc=False):
    """Try to compile a probe .c file. Returns True if the compiler supports this ISA."""
    with tempfile.NamedTemporaryFile(suffix=".obj" if is_msvc else ".o", delete=False) as tmp:
        obj_path = tmp.name
    try:
        prefix = [cc, "/c"] if is_msvc else [cc, "-c"]
        out_flag = ["/Fo" + obj_path] if is_msvc else ["-o", obj_path]
        return subprocess.run(prefix + flags + [probe_file] + out_flag, capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(obj_path)
        except OSError:
            pass


# (NK_TARGET_NAME, probe_file_relative_to_NK_ROOT, gcc_flags, msvc_flags)
PROBE_TABLE_X86 = [
    ("HASWELL",     "probes/x86_haswell.c",     ["-mavx2", "-mfma", "-mf16c"],                       ["/arch:AVX2"]),
    ("SKYLAKE",     "probes/x86_skylake.c",     ["-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl"], ["/arch:AVX512"]),
    ("ICELAKE",     "probes/x86_icelake.c",     ["-mavx512vnni", "-mavx512vl"],                       ["/arch:AVX512"]),
    ("GENOA",       "probes/x86_genoa.c",       ["-mavx512bf16", "-mavx512vl"],                       ["/arch:AVX512"]),
    ("SAPPHIRE",    "probes/x86_sapphire.c",    ["-mavx512fp16", "-mavx512vl"],                       ["/arch:AVX512"]),
    ("SAPPHIREAMX", "probes/x86_sapphireamx.c", ["-mamx-tile", "-mamx-int8"],                         ["/arch:AVX512"]),
    ("GRANITEAMX",  "probes/x86_graniteamx.c",  ["-mamx-tile", "-mamx-fp16"],                         ["/arch:AVX512"]),
    ("DIAMOND",     "probes/x86_diamond.c",     ["-mavx10.2-512"],                                    ["/arch:AVX10.2"]),
    ("TURIN",       "probes/x86_turin.c",       ["-mavx512vp2intersect"],                             ["/arch:AVX512"]),
    ("ALDER",       "probes/x86_alder.c",       ["-mavxvnni"],                                        ["/arch:AVX2"]),
    ("SIERRA",      "probes/x86_sierra.c",      ["-mavxvnniint8"],                                    ["/arch:AVX2"]),
]

PROBE_TABLE_ARM = [
    ("NEON",       "probes/arm_neon.c",          ["-march=armv8-a+simd"],                                    []),
    ("NEONHALF",   "probes/arm_neon_half.c",     ["-march=armv8.2-a+simd+fp16"],                             ["/arch:armv8.2"]),
    ("NEONSDOT",   "probes/arm_neon_sdot.c",     ["-march=armv8.2-a+dotprod"],                               ["/arch:armv8.4"]),
    ("NEONBFDOT",  "probes/arm_neon_bfdot.c",    ["-march=armv8.6-a+simd+bf16"],                             ["/arch:armv8.6"]),
    ("NEONFHM",    "probes/arm_neon_fhm.c",      ["-march=armv8.2-a+simd+fp16+fp16fml"],                     ["/arch:armv8.4"]),
    ("NEONFP8",    "probes/arm_neonfp8.c",       ["-march=armv8-a+simd+fp8dot4"],                            []),
    ("SVE",        "probes/arm_sve.c",           ["-march=armv8.2-a+sve"],                                   []),
    ("SVEHALF",    "probes/arm_sve_half.c",      ["-march=armv8.2-a+sve+fp16"],                              []),
    ("SVEBFDOT",   "probes/arm_sve_bfdot.c",     ["-march=armv8.2-a+sve+bf16"],                              []),
    ("SVESDOT",    "probes/arm_sve_sdot.c",      ["-march=armv8.2-a+sve+dotprod"],                           []),
    ("SVE2",       "probes/arm_sve2.c",          ["-march=armv8.2-a+sve2"],                                  []),
    ("SVE2P1",     "probes/arm_sve2p1.c",        ["-march=armv8.2-a+sve2p1"],                                []),
    ("SME",        "probes/arm_sme.c",           ["-march=armv8-a+sme"],                                     []),
    ("SME2",       "probes/arm_sme2.c",          ["-march=armv8-a+sme2"],                                    []),
    ("SME2P1",     "probes/arm_sme2p1.c",        ["-march=armv8-a+sme2p1"],                                  []),
    ("SMEF64",     "probes/arm_sme_f64.c",       ["-march=armv8-a+sme+sme-f64f64"],                          []),
    ("SMEHALF",    "probes/arm_sme_half.c",      ["-march=armv8-a+sme+sme-f16f16"],                          []),
    ("SMEBF16",    "probes/arm_sme_bf16.c",      ["-march=armv8-a+sme2+b16b16"],                             []),
    ("SMEBI32",    "probes/arm_sme_bi32.c",      ["-march=armv8-a+sme2+sme-i16i32"],                         []),
    ("SMELUT2",    "probes/arm_sme_lut2.c",      ["-march=armv8-a+sme2+lut"],                                []),
    ("SMEFA64",    "probes/arm_sme_fa64.c",      ["-march=armv8-a+sme+sme-fa64"],                            []),
]

PROBE_TABLE_RISCV = [
    ("RVV",      "probes/riscv_rvv.c",       ["-march=rv64gcv"],            []),
    ("RVVHALF",  "probes/riscv_rvv_half.c",  ["-march=rv64gcv_zvfh"],       []),
    ("RVVBF16",  "probes/riscv_rvv_bf16.c",  ["-march=rv64gcv_zvfbfwma"],   []),
    ("RVVBB",    "probes/riscv_rvv_bb.c",    ["-march=rv64gcv_zvbb"],       []),
]

PROBE_TABLE_LOONGARCH = [
    ("LOONGSONASX", "probes/loongarch_lasx.c", ["-mlasx"], []),
]

PROBE_TABLE_POWER = [
    ("POWERVSX", "probes/power_vsx.c", ["-mcpu=power9", "-mvsx"], []),
]

PROBE_TABLE_WASM = [
    ("V128RELAXED", "probes/wasm_v128relaxed.c", ["-mrelaxed-simd"], []),
]


def probe_all_isas():
    cc, is_msvc = detect_cc()
    macros = []
    tables = [
        (is_64bit_x86(),       PROBE_TABLE_X86),
        (is_64bit_arm(),       PROBE_TABLE_ARM),
        (is_64bit_riscv(),     PROBE_TABLE_RISCV),
        (is_64bit_loongarch(), PROBE_TABLE_LOONGARCH),
        (is_64bit_power(),     PROBE_TABLE_POWER),
        (is_wasm(),            PROBE_TABLE_WASM),
    ]
    for arch_match, table in tables:
        for name, probe_rel, gcc_flags, msvc_flags in table:
            probe_path = os.path.join(NK_ROOT, probe_rel)
            if arch_match and os.path.isfile(probe_path):
                env_val = os.environ.get(f"NK_TARGET_{name}")
                if env_val == "0":
                    macros.append((f"NK_TARGET_{name}", "0"))
                    continue
                flags = msvc_flags if is_msvc else gcc_flags
                ok = probe_isa(cc, probe_path, flags, is_msvc)
                macros.append((f"NK_TARGET_{name}", "1" if ok else "0"))
                print(f"[colbandit] Probe NK_TARGET_{name}: {'supported' if ok else 'not supported'}")
            else:
                macros.append((f"NK_TARGET_{name}", "0"))
    return macros


# ---------------------------------------------------------------------------
# Per-platform compile/link flags (lifted from upstream)
# ---------------------------------------------------------------------------
def linux_settings():
    compile_args = ["-std=c11", "-O3", "-fdiagnostics-color=always",
                    "-fvisibility=default", "-fPIC", "-w", "-fopenmp"]
    if is_64bit_riscv():
        compile_args.append("-march=rv64gcv")
    link_args = ["-shared", "-lm", "-lgomp"]
    macros = [("NK_DYNAMIC_DISPATCH", "1"), ("NK_NATIVE_F16", "0"), ("NK_NATIVE_BF16", "0")]
    macros.extend(probe_all_isas())
    return compile_args, link_args, macros


def darwin_settings():
    compile_args = ["-std=c11", "-O3", "-w"]
    link_args: list[str] = []
    try:
        libomp_prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL
        ).decode().strip()
        compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{libomp_prefix}/include"]
        link_args += [f"-L{libomp_prefix}/lib", "-lomp"]
    except Exception:
        compile_args.append("-DNK_NO_OPENMP=1")
        print("[colbandit] WARNING: libomp not found (brew install libomp). OpenMP disabled.")
    macros = [("NK_DYNAMIC_DISPATCH", "1"), ("NK_NATIVE_F16", "0"), ("NK_NATIVE_BF16", "0")]
    macros.extend(probe_all_isas())
    return compile_args, link_args, macros


def freebsd_settings():
    compile_args = ["-std=c11", "-O3", "-fdiagnostics-color=always",
                    "-fvisibility=default", "-fPIC", "-w"]
    link_args = ["-shared", "-lm"]
    macros = [("NK_DYNAMIC_DISPATCH", "1"), ("NK_NATIVE_F16", "0"), ("NK_NATIVE_BF16", "0")]
    macros.extend(probe_all_isas())
    return compile_args, link_args, macros


def windows_settings():
    compile_args = ["/std:c11", "/O2", "/d2FH4-", "/w"]
    link_args: list[str] = []
    macros = [("NK_DYNAMIC_DISPATCH", "1"), ("NK_NATIVE_F16", "0"), ("NK_NATIVE_BF16", "0")]
    macros.extend(probe_all_isas())
    if is_64bit_arm():
        macros.append(("_ARM64_", "1"))
    elif is_64bit_x86():
        macros.append(("_AMD64_", "1"))
    return compile_args, link_args, macros


def emscripten_settings():
    compile_args = ["-std=c11", "-O3", "-w"]
    link_args: list[str] = []
    macros = [("NK_DYNAMIC_DISPATCH", "1"), ("NK_PYODIDE_SIDE_MODULE", "1"),
              ("NK_NATIVE_F16", "0"), ("NK_NATIVE_BF16", "0")]
    macros.extend(probe_all_isas())
    return compile_args, link_args, macros


_host_platform = os.environ.get("_PYTHON_HOST_PLATFORM", "")
if "emscripten" in _host_platform:
    compile_args, link_args, macros = emscripten_settings()
elif sys.platform == "linux":
    compile_args, link_args, macros = linux_settings()
elif sys.platform.startswith("freebsd"):
    compile_args, link_args, macros = freebsd_settings()
elif sys.platform == "darwin":
    compile_args, link_args, macros = darwin_settings()
elif sys.platform == "win32":
    compile_args, link_args, macros = windows_settings()
else:
    compile_args, link_args, macros = [], [], []


# ---------------------------------------------------------------------------
# Source list — same as native/numkong/setup.py, paths rebased to NK_ROOT
# ---------------------------------------------------------------------------
base_sources = [
    f"{NK_ROOT}/python/numkong.c",
    f"{NK_ROOT}/python/tensor.c",
    f"{NK_ROOT}/python/matrix.c",
    f"{NK_ROOT}/python/types.c",
    f"{NK_ROOT}/python/distance.c",
    f"{NK_ROOT}/python/each.c",
    f"{NK_ROOT}/python/mesh.c",
    f"{NK_ROOT}/python/maxsim.c",
    f"{NK_ROOT}/python/colbandit.c",
    f"{NK_ROOT}/python/numpy_interop.c",
    f"{NK_ROOT}/c/numkong.c",
]
dispatch_sources = sorted(glob.glob(f"{NK_ROOT}/c/dispatch_*.c"))

ext_modules = [
    Extension(
        "colbandit._kernel",
        sources=base_sources + dispatch_sources,
        include_dirs=[f"{NK_ROOT}/include", f"{NK_ROOT}/python"],
        language="c",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros,
    )
]


class ParallelBuildExt(build_ext):
    """Limit parallel C compilation to keep cibuildwheel containers from OOMing."""

    def initialize_options(self):
        super().initialize_options()
        self.parallel = int(os.environ.get("NK_BUILD_PARALLEL", min(os.cpu_count() or 1, 4)))


setup(
    cmdclass={"build_ext": ParallelBuildExt},
    ext_modules=ext_modules,
    zip_safe=False,
)
