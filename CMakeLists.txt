cmake_minimum_required(VERSION 3.20)
project(neuromancers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)

add_executable(neuromancers
        main.cpp
        include/types.hpp
        include/edt.hpp
        src/edt.cpp
        include/dijkstra.hpp
        src/dijkstra.cpp
        include/teasar.hpp
        src/teasar.cpp
        include/rdp.hpp
        src/rdp.cpp
        include/swc.hpp
        src/swc.cpp
        include/viewer.hpp
        src/viewer.cpp
        include/volume_io.hpp
        src/volume_io.cpp
        include/mesh.hpp
        src/mesh.cpp
        include/mesh_teasar.hpp
        src/mesh_teasar.cpp
        include/voxelizer.hpp
        src/voxelizer.cpp
)

target_include_directories(neuromancers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (OpenMP_CXX_FOUND)
    target_link_libraries(neuromancers PRIVATE OpenMP::OpenMP_CXX)
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(neuromancers PRIVATE -Wall -Wextra -Wpedantic)
endif ()
