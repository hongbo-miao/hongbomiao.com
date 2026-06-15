import ctypes.util
from pathlib import Path

# Homebrew installs libusb outside the default macOS dynamic linker search path,
# so pyusb's ctypes lookup cannot find it without help.
_HOMEBREW_LIBUSB_CANDIDATE_LIST: tuple[str, ...] = (
    "/opt/homebrew/lib/libusb-1.0.dylib",
    "/usr/local/lib/libusb-1.0.dylib",
)

_is_configured: bool = False


def configure_libusb() -> None:
    """Patch ctypes library resolution so pyusb finds a Homebrew libusb.

    Idempotent; safe to call before any pyftdi access.
    """
    global _is_configured
    if _is_configured:
        return

    original_find_library = ctypes.util.find_library

    def find_library_with_homebrew_fallback(name: str) -> str | None:
        found: str | None = original_find_library(name)
        if found is not None:
            return found
        if name in ("usb-1.0", "usb"):
            for candidate in _HOMEBREW_LIBUSB_CANDIDATE_LIST:
                if Path(candidate).exists():
                    return candidate
        return found

    ctypes.util.find_library = find_library_with_homebrew_fallback
    _is_configured = True
