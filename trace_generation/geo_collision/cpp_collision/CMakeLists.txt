cmake_minimum_required(VERSION 3.12)
project(cpp_collision)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 优化标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

# 查找 Python 和 pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 尝试使用 Python 模块方式查找 pybind11
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(pybind11_DIR)
    find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
else()
    # 如果上面失败，尝试直接获取头文件路径
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE pybind11_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # 手动设置 pybind11
    add_library(pybind11::module INTERFACE IMPORTED)
    set_target_properties(pybind11::module PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${pybind11_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES Python3::Module
    )
endif()

# 创建 Python 扩展模块
if(TARGET pybind11::module)
    pybind11_add_module(cpp_collision bindings.cpp)
else()
    # 手动创建扩展模块
    Python3_add_library(cpp_collision MODULE bindings.cpp)
    target_link_libraries(cpp_collision PRIVATE pybind11::module)
    set_target_properties(cpp_collision PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
endif()

# 设置输出目录
set_target_properties(cpp_collision PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# 包含头文件目录
target_include_directories(cpp_collision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 安装规则
install(TARGETS cpp_collision
    LIBRARY DESTINATION .
)
