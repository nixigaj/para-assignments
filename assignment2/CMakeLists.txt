add_executable(ass2_exe1 exercise1/main.c)
add_executable(ass2_exe2 exercise2/main.c)

target_link_libraries(ass2_exe1 m)
target_link_libraries(ass2_exe2 m)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
	target_compile_options(ass2_exe1 PRIVATE -march=native -O3 -ffast-math)
	target_compile_options(ass2_exe2 PRIVATE -march=native -O3 -ffast-math)
endif()
