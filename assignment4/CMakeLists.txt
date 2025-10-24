find_package(MPI REQUIRED)

add_executable(ass4_exe0 exercise0/hello_mpi.c)
add_executable(ass4_exe1_send_recv exercise1/send_recv.c)
add_executable(ass4_exe1_bcast_reduce exercise1/bcast_reduce.c)

target_link_libraries(ass4_exe0 m MPI::MPI_C)
target_link_libraries(ass4_exe1_send_recv m MPI::MPI_C)
target_link_libraries(ass4_exe1_bcast_reduce m MPI::MPI_C)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
	target_compile_options(ass4_exe0 PRIVATE -march=native -O3 -ffast-math)
	target_compile_options(ass4_exe1_send_recv PRIVATE -march=native -O3 -ffast-math)
	target_compile_options(ass4_exe1_bcast_reduce PRIVATE -march=native -O3 -ffast-math)
endif()
