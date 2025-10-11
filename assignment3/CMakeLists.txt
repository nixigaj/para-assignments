find_package(OpenMP REQUIRED)

set(COMMON_FLAGS -std=c11)

add_executable(ass3_exe1     exercise1/main.c)
add_executable(ass3_exe2     exercise2/Game_Of_Life_OpenMP.c)
add_executable(ass3_exe2_seq exercise2/Game_Of_Life.c)
#add_executable(ass3_exe3     exercise3/main.c)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	target_compile_options(ass3_exe1     PRIVATE ${COMMON_FLAGS} -g -O0 -fopenmp)
	target_compile_options(ass3_exe2     PRIVATE ${COMMON_FLAGS} -g -O0 -fopenmp)
	target_compile_options(ass3_exe2_seq PRIVATE ${COMMON_FLAGS} -g -O0)
	#target_compile_options(ass3_exe3     PRIVATE ${COMMON_FLAGS} -g -O0 -fopenmp)

	target_link_libraries(ass3_exe1     PRIVATE m OpenMP::OpenMP_C)
	target_link_libraries(ass3_exe2     PRIVATE m OpenMP::OpenMP_C)
	target_link_libraries(ass3_exe2_seq PRIVATE m)
	#target_link_libraries(ass3_exe3     PRIVATE m OpenMP::OpenMP_C)

	target_compile_definitions(ass3_exe2     PRIVATE OUTPUT)
	target_compile_definitions(ass3_exe2_seq PRIVATE OUTPUT)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
	set(RELEASE_OPTS -O3 -march=native -ffast-math)

	target_compile_options(ass3_exe1     PRIVATE ${COMMON_FLAGS} ${RELEASE_OPTS} -fopenmp)
	target_compile_options(ass3_exe2     PRIVATE ${COMMON_FLAGS} ${RELEASE_OPTS} -fopenmp)
	target_compile_options(ass3_exe2_seq PRIVATE ${COMMON_FLAGS} ${RELEASE_OPTS})
	#target_compile_options(ass3_exe3     PRIVATE ${COMMON_FLAGS} ${RELEASE_OPTS} -fopenmp)

	target_link_libraries(ass3_exe1     PRIVATE m OpenMP::OpenMP_C)
	target_link_libraries(ass3_exe2     PRIVATE m OpenMP::OpenMP_C)
	target_link_libraries(ass3_exe2_seq PRIVATE m)
	#target_link_libraries(ass3_exe3     PRIVATE m OpenMP::OpenMP_C)
endif()
