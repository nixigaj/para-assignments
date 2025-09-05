# Introduction to Paralell Programming assignments

To run the release build on a UNIX-like OS:
```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./assignmentX/assX_exeY
```

Where `X` and `Y` are replaced with the assignment and exercise number.

For Windows, you need CMake (`winget install -e --id Kitware.CMake`) and Visual Studio C++ tools.

To run the release build on Windows, run:
```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\assignmentX\Release\assX_exeY.exe
```

Where `X` and `Y` are replaced with the assignment and exercise number.