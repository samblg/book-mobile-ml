#ifndef PLATFORM_H
#define PLATFORM_H

//CPU
//__CPU_X86__: Intel x86 cpu
//__CPU_X64__: Intel amd64 cpu
//__CPU_ARM__: ARM cpu
//__CPU_AARCH64__: ARM ARCH64 cpu
//__CPU_MIPS__: MIPS cpu
#if defined(i386)
#define __CPU_X86__
#elif defined(_M_IX86)
#define __CPU_X86__
#elif defined(__x86_64__)
#define __CPU_X64__
#elif defined(_M_X64)
#define __CPU_X64__
#elif defined(__CPU_ARM__)
#else
#error "Can't detect the CPU!"
#endif //CPU

// Operation System:
// 	__OS_WINDOWS__: Windows
// 	__OS_UNIX__: Linux and other *nix
#if defined(_WIN32) | defined(_WIN64)
#define __OS_WINDOWS__
#elif defined(__unix__)
#define __OS_UNIX__
#elif defined(__APPLE__)
#define __OS_UNIX__
#else
#error "Can't detect the Operating System"
#endif //OS

// Compiler:
// __COMPILER_GCC__: GCC
// __COMPILER_VC__: Visual Studio
// __COMPILER_CLANG__: Clang
// (clang also define __GNUC__, so we need put __clang_version__ on top)
#if defined(__clang_version__)
#define __COMPILER_CLANG__
#elif defined(__GNUC__)
#define __COMPILER_GCC__
#elif defined(_MSC_VER)
#define __COMPILER_MSVC__
#else
#error "Can't detect the Compiler"
#endif // Compiler

#endif // PLATFORM_H
