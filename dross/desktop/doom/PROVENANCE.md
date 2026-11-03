# DoomGeneric

Upstream: https://github.com/ozkl/doomgeneric
Revision: e257a09527062bfb986b6e5e0780839b6cbd5b71
Fetched from the GitHub source archive over HTTPS on 2026-11-03.

The retained upstream C files and headers are unmodified. Unused platform adapters
are omitted. Original README, Makefile, file copyright/license headers and GPLv2
LICENSE are retained. The combined oscar-doom executable is GPL-2.0-or-later;
Oscar window protocol and separately linked services keep their own licenses.
The engine's software rasterizer is unchanged. The Rust adapter replaces the
platform callbacks with native Oscar window channels/shared surfaces and keyboard
state. No proprietary IWAD is included. Freedoom data is fetched separately by the
build recipe with its own redistribution notices.
