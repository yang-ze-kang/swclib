"""Kernel-enforced read-only access for Linux HTTP threads and SWC processes.

The server opens source files only in protected readers, after installing the
filter. No writable source descriptors are handed to them. Restrictions also
cover native TIFF/NRRD libraries and newly created decoder threads. The main
thread, browser launcher, stdout and HTTP sockets remain usable.

libseccomp ABI: https://github.com/seccomp/libseccomp/blob/main/include/seccomp.h.in
"""

import ctypes
import errno
import os
import sys
from types import SimpleNamespace


def prepare_scipy_memory_stream():
    """Keep Qhull diagnostic streams in anonymous memory under seccomp.

    Some SciPy builds lack open_memstream and use mkstemp + fdopen instead.
    Replace only messagestream's two module-local I/O providers, not the global
    os/tempfile modules. The fallback gets an anonymous descriptor; there is no
    filesystem path to create or unlink, and source-write rules stay unchanged.
    See scipy/_lib/messagestream.pyx (SciPy 1.13.1).
    """
    import scipy._lib.messagestream as streams
    marker = b"swclib-anonymous-qhull-stream"
    # Older Python builds omit os.memfd_create even on a modern Linux host.
    libc = ctypes.CDLL(None, use_errno=True)
    create = libc.memfd_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int

    def temporary(*args, **kwargs):
        descriptor = create(b"swclib-qhull", 1)  # MFD_CLOEXEC
        if descriptor < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return descriptor, marker

    def remove(path):
        if path is not None and path != marker:
            raise PermissionError("Only anonymous Qhull streams may be released")

    streams.tempfile = SimpleNamespace(mkstemp=temporary)
    streams.os = SimpleNamespace(remove=remove, close=os.close)


class ArgCompare(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


class ReadOnlyGuard:
    def __init__(self):
        if sys.platform != "linux":
            raise RuntimeError("只读文件保护需要 Linux 和 libseccomp；未启用保护时不启动数据服务")
        try:
            self.lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
        except OSError as exc:
            raise RuntimeError("无法加载 libseccomp.so.2，只读保护未就绪；请安装系统 libseccomp2") from exc
        lib = self.lib
        lib.seccomp_init.argtypes = [ctypes.c_uint32]
        lib.seccomp_init.restype = ctypes.c_void_p
        lib.seccomp_release.argtypes = [ctypes.c_void_p]
        lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
        lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
        lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                              ctypes.c_uint, ctypes.POINTER(ArgCompare)]
        lib.seccomp_rule_add_array.restype = ctypes.c_int
        lib.seccomp_load.argtypes = [ctypes.c_void_p]
        lib.seccomp_load.restype = ctypes.c_int

    def install(self):
        """Irreversibly restrict the calling thread/worker; fail closed."""
        lib = self.lib
        context = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
        if not context:
            raise RuntimeError("无法创建只读文件保护")

        def deny(name, argument=None, flag=None):
            number = lib.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number == -1:  # Not known on this architecture/libseccomp version.
                return
            comparison = ArgCompare(argument, 7, flag, flag) if argument is not None else None
            result = lib.seccomp_rule_add_array(context, 0x00050000 | errno.EACCES,
                                                number, int(comparison is not None),
                                                ctypes.byref(comparison) if comparison is not None else None)
            if result < 0:
                raise RuntimeError(f"无法安装只读规则 {name}: {os.strerror(-result)}")

        try:
            flags = [os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_TRUNC, os.O_APPEND]
            if hasattr(os, "O_TMPFILE"):
                flags.append(os.O_TMPFILE & ~os.O_DIRECTORY)
            for name, argument in (("open", 1), ("openat", 2)):
                for flag in flags:
                    deny(name, argument, flag)
            # openat2 stores flags behind a pointer, so reject the syscall entirely.
            # Ordinary read-only open/openat remain available to all current readers.
            for name in (
                "openat2", "creat", "open_by_handle_at", "truncate", "truncate64", "ftruncate", "ftruncate64",
                "fallocate", "unlink", "unlinkat", "rmdir", "rename", "renameat", "renameat2",
                "mkdir", "mkdirat", "link", "linkat", "symlink", "symlinkat", "mknod", "mknodat",
                "chmod", "fchmod", "fchmodat", "fchmodat2", "chown", "lchown", "fchown", "fchownat",
                "chown32", "lchown32", "fchown32", "utime", "utimes", "futimesat", "utimensat",
                "setxattr", "lsetxattr", "fsetxattr", "removexattr", "lremovexattr", "fremovexattr",
                "mount", "umount2", "pivot_root", "io_uring_setup", "io_submit", "ioctl",
            ):
                deny(name)
            result = lib.seccomp_load(context)
            if result < 0:
                raise RuntimeError(f"只读文件保护启动失败: {os.strerror(-result)}")
        finally:
            lib.seccomp_release(context)
