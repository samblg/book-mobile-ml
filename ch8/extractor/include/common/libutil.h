#pragma once

#include <string>
#include <iostream>

#ifdef WIN32
const char PATH_SLASH = '\\';
#else
const char PATH_SLASH = '/';
#endif // WIN32

#ifdef WIN32
#include <Windows.h>

typedef HMODULE LibraryHandle;

LibraryHandle AmLibraryLoad(const std::string& name);
#define AmLibraryFree FreeLibrary
inline DWORD AmGetLibraryError() {
    return GetLastError();
}

#else // WIN32

#include <dlfcn.h>

typedef void* LibraryHandle;

LibraryHandle AmLibraryLoad(const std::string& name);
#define AmLibraryFree dlclose
#define AmGetLibraryError() dlerror()

#endif // WIN32

template <class Function>
Function AmLibraryGetSymbol(LibraryHandle libraryHandle, const std::string& libraryName) {
#ifdef WIN32
    return reinterpret_cast<Function>(GetProcAddress(libraryHandle, libraryName.c_str()));
#else
    return reinterpret_cast<Function>(dlsym(libraryHandle, libraryName.c_str()));
#endif
}

std::string GetLibraryPath();
