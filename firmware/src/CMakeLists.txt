# This file was automatically generated for projects
# without default 'CMakeLists.txt' file.

FILE(GLOB_RECURSE app_sources ${CMAKE_SOURCE_DIR}/src/*.*)

idf_component_register(SRCS ${app_sources})

# The sdkconfig of this project uses CONFIG_COMPILER_OPTIMIZATION_LEVEL_DEBUG (-Og
# global). The components espressif__esp-tflite-micro (-O3) and espressif__esp-nn
# (-O2) override it through target_compile_options in their own CMakeLists,
# but the "src" component (main.cpp) does NOT -- proven in compile_commands.json
# main.cpp only gets "-Og". Yet computeLogDensity() = ~92% of the latency per
# task (313 ms). Force -O2 for all components here; the last flag in
# command line menang atas -Og bawaan IDF.
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
