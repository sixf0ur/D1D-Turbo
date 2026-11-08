@echo off
rem Builds drumgen.exe (CLI) and example.exe (library demo) with MSVC (Visual Studio
rem 2019+ or Build Tools, "Desktop development with C++"). Static runtime (/MT): the
rem .exe files need no DLLs besides Windows itself.
rem   build.bat            AVX2 build (any x86-64 CPU from ~2013 on)
rem   build.bat portable   no SIMD intrinsics (older CPUs; same results, ~5x slower)
setlocal
cd /d "%~dp0"
set ARCH=/arch:AVX2
if /i "%1"=="portable" set ARCH=/DD1D_NO_SIMD

where cl >nul 2>nul && goto :build
rem not in a Developer Prompt: find Visual Studio with vswhere and load its environment
set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "%VSWHERE%" goto :novs
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSDIR=%%i
if not defined VSDIR goto :novs
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || goto :novs

:build
cl /nologo /O2 %ARCH% /std:c++17 /EHsc /MT /W3 drumgen.cpp /Fe:drumgen.exe || exit /b 1
cl /nologo /O2 %ARCH% /std:c++17 /EHsc /MT /W3 example.cpp /Fe:example.exe || exit /b 1
del /q drumgen.obj example.obj 2>nul
echo built %~dp0drumgen.exe and %~dp0example.exe
exit /b 0

:novs
echo Visual Studio C++ tools not found. Install "Build Tools for Visual Studio" with the
echo "Desktop development with C++" workload, or run this from a Developer Command Prompt.
exit /b 1
