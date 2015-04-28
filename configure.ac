#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([myriad], [1.0], [pedro.rittner@outlook.com])
AC_CONFIG_SRCDIR([src/points.c])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CC
AC_PROG_MAKE_SET

# Set C99 flag
AC_PROG_CC_C99

# Checks for libraries.
# FIXME: Replace `main' with a function in `-lmyriad':
#AC_CHECK_LIB([myriad], [main])

# Checks for header files.
AC_CHECK_HEADERS([stddef.h stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_REALLOC

AC_CONFIG_FILES([Makefile
                 src/Makefile])
AC_OUTPUT
