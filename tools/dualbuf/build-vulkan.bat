@echo off
rem Laptop (Windows) Release build: CPU + Vulkan, native MSVC toolchain.
rem Requires: VS 2022 Community (C++ workload, CMake, Ninja) and the Vulkan SDK.
set "VS=C:\Program Files\Microsoft Visual Studio\2022\Community"
if "%VULKAN_SDK%"=="" set "VULKAN_SDK=C:\VulkanSDK\1.4.363.1"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "PATH=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
cd /d "%~dp0..\.."
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DLLAMA_CURL=OFF || exit /b 1
cmake --build build --config Release -j 6 || exit /b 1
echo BUILD_OK
