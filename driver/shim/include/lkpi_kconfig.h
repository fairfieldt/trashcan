/* SPDX-License-Identifier: MIT */
/* IS_ENABLED() for options that are either defined to 1 or not defined at all. */
#ifndef GPUD_LKPI_KCONFIG_H
#define GPUD_LKPI_KCONFIG_H
#define __LKPI_PLACEHOLDER_1 0,
#define __lkpi_second(ignored, val, ...) val
#define __lkpi_defined(x) ___lkpi_defined(x)
#define ___lkpi_defined(val) ____lkpi_defined(__LKPI_PLACEHOLDER_##val)
#define ____lkpi_defined(arg1_or_junk) __lkpi_second(arg1_or_junk 1, 0)
#define IS_ENABLED(option) __lkpi_defined(option)
#define IS_BUILTIN(option) __lkpi_defined(option)
#define IS_REACHABLE(option) __lkpi_defined(option)
#endif
