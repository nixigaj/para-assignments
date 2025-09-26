# Introduction to Parallel Programming assignments

The assignments are tested on the Uppsalas's Linux servers.

To run the release build, you need CMake, Make, and a C11/C++20 compiler in your path, then run
```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./assignmentX/assX_exeY
```

where `X` and `Y` are replaced with the assignment and exercise number.
