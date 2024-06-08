#pragma once

//CPU
//__CPU_X86__: Intel x86 cpu
//__CPU_X64__: Intel amd64 cpu
//__CPU_ARM__: ARM cpu
//__CPU_AARCH64__: ARM ARCH64 cpu
//__CPU_MIPS__: MIPS cpu
#if (defined(i386) || defined(_M_IX86))
#if !defined(__CPU_X86__)
#define __CPU_X86__
#endif // __CPU_X86__
#elif defined(__arm__)
#if !defined(__CPU_ARM__)
#define __CPU_ARM__
#endif // __CPU_ARM__
#elif defined(__aarch64__)
#if !defined(__CPU_AARCH64__)
#define __CPU_AARCH64__
#endif // __CPU_AARCH64__
#elif defined(__x86_64__) || defined(_M_X64)
#if !defined(__CPU_X64__)
#define __CPU_X64__
#endif // __CPU_X64__
#else
#error "Can't detect the CPU!"
#endif //CPU

// Operation System:
// 	__OS_WINDOWS__: Windows
// 	__OS_UNIX__: Linux and other *nix
#if defined(_WIN32) || defined(_WIN64)
#if !defined(__OS_WINDOWS__)
#define __OS_WINDOWS__
#endif // __OS__WINDOWS__
#elif defined(__unix__)
#if !defined(__OS_UNIX__)
#define __OS_UNIX__
#endif // __OS_UNIX__
#elif defined(__APPLE__)
#if !defined(__OS_UNIX__)
#define __OS_UNIX__
#endif // __OS_UNIX__
#else
#error "Can't detect the Operating System"
#endif //OS

// Compiler:
// __COMPILER_GCC__: GCC
// __COMPILER_VC__: Visual Studio
// __COMPILER_CLANG__: Clang
#if defined(__GNUC__)
#if !defined(__COMPILER_GCC__)
#define __COMPILER_GCC__
#endif // __COMPILER_GCC__
#elif defined(_MSC_VER)
#if !defined(__COMPILER_MSVC__)
#define __COMPILER_MSVC__
#endif // __COMPILER_MSVC__
#elif defined(__clang_version__)
#if !defined(__COMPILER_CLANG__)
#define __COMPILER_CLANG__
#endif // __COMPILER_CLANG__
#else
#error "Can't detect the Compiler"
#endif // Compiler
