# Introduction to Paralell Programming assignments

The assignments are tested on the Uppsalas's Linux servers and Windows 11.

To run the release build on a UNIX-like OS, you need CMake, Make, and a C++ compiler in your path, then run
```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./assignmentX/assX_exeY
```

where `X` and `Y` are replaced with the assignment and exercise number.

For Windows, you need CMake (`winget install -e --id Kitware.CMake`) and Visual Studio C++ tools.

To run the release build on Windows, run
```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\assignmentX\Release\assX_exeY.exe
```

where `X` and `Y` are replaced with the assignment and exercise number.