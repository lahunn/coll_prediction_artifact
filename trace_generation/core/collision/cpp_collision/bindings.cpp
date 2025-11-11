#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "collision_detection.h"
#include "sphere_collision_checker.h"

namespace py = pybind11;
using namespace collision;

PYBIND11_MODULE(cpp_collision, m) {
    m.doc() = "C++ 加速版碰撞检测库";

    // 绑定 Sphere 类
    py::class_<Sphere>(m, "Sphere")
        .def(py::init<double, double, double, double>(),
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("r"))
        .def_readwrite("x", &Sphere::x)
        .def_readwrite("y", &Sphere::y)
        .def_readwrite("z", &Sphere::z)
        .def_readwrite("r", &Sphere::r)
        .def_readwrite("r_sq", &Sphere::r_sq);

    // 绑定 AABB 类
    py::class_<AABB>(m, "AABB")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("min_x"), py::arg("min_y"), py::arg("min_z"),
             py::arg("max_x"), py::arg("max_y"), py::arg("max_z"))
        .def_readwrite("min_x", &AABB::min_x)
        .def_readwrite("min_y", &AABB::min_y)
        .def_readwrite("min_z", &AABB::min_z)
        .def_readwrite("max_x", &AABB::max_x)
        .def_readwrite("max_y", &AABB::max_y)
        .def_readwrite("max_z", &AABB::max_z);

    // 绑定 Cuboid 类
    py::class_<Cuboid>(m, "Cuboid")
        .def(py::init<double, double, double,
                      double, double, double, double,
                      double, double, double, double,
                      double, double, double, double>(),
             py::arg("x"), py::arg("y"), py::arg("z"),
             py::arg("axis_1_x"), py::arg("axis_1_y"), py::arg("axis_1_z"), py::arg("axis_1_r"),
             py::arg("axis_2_x"), py::arg("axis_2_y"), py::arg("axis_2_z"), py::arg("axis_2_r"),
             py::arg("axis_3_x"), py::arg("axis_3_y"), py::arg("axis_3_z"), py::arg("axis_3_r"))
        .def_readwrite("x", &Cuboid::x)
        .def_readwrite("y", &Cuboid::y)
        .def_readwrite("z", &Cuboid::z);

    // 绑定 Capsule 类
    py::class_<Capsule>(m, "Capsule")
        .def(py::init<double, double, double, double, double, double, double>(),
             py::arg("x1"), py::arg("y1"), py::arg("z1"),
             py::arg("xv"), py::arg("yv"), py::arg("zv"), py::arg("r"))
        .def_readwrite("x1", &Capsule::x1)
        .def_readwrite("y1", &Capsule::y1)
        .def_readwrite("z1", &Capsule::z1)
        .def_readwrite("xv", &Capsule::xv)
        .def_readwrite("yv", &Capsule::yv)
        .def_readwrite("zv", &Capsule::zv)
        .def_readwrite("r", &Capsule::r);

    // 绑定 HeightField 类
    py::class_<HeightField>(m, "HeightField")
        .def(py::init<double, double, double, double, double, double, int, int, const std::vector<double>&>(),
             py::arg("x"), py::arg("y"), py::arg("z"),
             py::arg("xs"), py::arg("ys"), py::arg("zs"),
             py::arg("xd"), py::arg("yd"), py::arg("data"))
        .def_readwrite("x", &HeightField::x)
        .def_readwrite("y", &HeightField::y)
        .def_readwrite("z", &HeightField::z);

    // 绑定 Triangle 类
    py::class_<Triangle>(m, "Triangle")
        .def(py::init<double, double, double, double, double, double, double, double, double>(),
             py::arg("v0_x"), py::arg("v0_y"), py::arg("v0_z"),
             py::arg("v1_x"), py::arg("v1_y"), py::arg("v1_z"),
             py::arg("v2_x"), py::arg("v2_y"), py::arg("v2_z"))
        .def_readwrite("v0_x", &Triangle::v0_x)
        .def_readwrite("v0_y", &Triangle::v0_y)
        .def_readwrite("v0_z", &Triangle::v0_z)
        .def_readwrite("v1_x", &Triangle::v1_x)
        .def_readwrite("v1_y", &Triangle::v1_y)
        .def_readwrite("v1_z", &Triangle::v1_z)
        .def_readwrite("v2_x", &Triangle::v2_x)
        .def_readwrite("v2_y", &Triangle::v2_y)
        .def_readwrite("v2_z", &Triangle::v2_z);

    // 绑定碰撞检测函数
    m.def("sphere_sphere", &sphere_sphere,
          "球-球碰撞检测\n\n"
          "Args:\n"
          "    sphere_a: 第一个球体\n"
          "    sphere_b: 第二个球体\n\n"
          "Returns:\n"
          "    int: 1=无碰撞, 0=碰撞",
          py::arg("sphere_a"), py::arg("sphere_b"));

    m.def("sphere_aabb", &sphere_aabb,
          "球-AABB碰撞检测\n\n"
          "Args:\n"
          "    sphere: 球体\n"
          "    aabb: 轴对齐包围盒\n\n"
          "Returns:\n"
          "    tuple: (collision_result, cycles)\n"
          "           collision_result: 1=无碰撞, 0=碰撞\n"
          "           cycles: 硬件周期数",
          py::arg("sphere"), py::arg("aabb"));

    m.def("cuboid_sphere", &cuboid_sphere,
          "OBB-球碰撞检测\n\n"
          "Args:\n"
          "    cuboid: 有向包围盒\n"
          "    sphere: 球体\n\n"
          "Returns:\n"
          "    int: 1=无碰撞, 0=碰撞",
          py::arg("cuboid"), py::arg("sphere"));

    m.def("sphere_cuboid", &sphere_cuboid,
          "球-OBB碰撞检测（cuboid_sphere的别名）",
          py::arg("cuboid"), py::arg("sphere"));

    m.def("cuboid_aabb", &cuboid_aabb,
          "AABB-OBB碰撞检测 (SAT算法)\n\n"
          "Args:\n"
          "    cuboid: 有向包围盒\n"
          "    aabb: 轴对齐包围盒\n\n"
          "Returns:\n"
          "    tuple: (collision_result, cycles)\n"
          "           collision_result: 1=无碰撞, 0=碰撞\n"
          "           cycles: 硬件周期数",
          py::arg("cuboid"), py::arg("aabb"));

    m.def("sphere_capsule", &sphere_capsule,
          "球-胶囊碰撞检测",
          py::arg("capsule"), py::arg("sphere"));

    m.def("cuboid_capsule", &cuboid_capsule,
          "OBB-胶囊碰撞检测",
          py::arg("cuboid"), py::arg("capsule"));

    m.def("sphere_heightfield", &sphere_heightfield,
          "球-高度场碰撞检测",
          py::arg("heightfield"), py::arg("sphere"));

    m.def("cuboid_cuboid", &cuboid_cuboid,
          "OBB-OBB碰撞检测",
          py::arg("cuboid_a"), py::arg("cuboid_b"));

    m.def("cuboid_heightfield", &cuboid_heightfield,
          "OBB-高度场碰撞检测",
          py::arg("cuboid"), py::arg("heightfield"));

    m.def("sphere_triangle", &sphere_triangle,
          "球-三角形碰撞检测",
          py::arg("sphere"), py::arg("triangle"));

    m.def("cuboid_triangle", &cuboid_triangle,
          "OBB-三角形碰撞检测",
          py::arg("cuboid"), py::arg("triangle"));

    // ========================================================================
    // 绑定 SphereCollisionChecker 类
    // ========================================================================
    py::class_<SphereCollisionChecker>(m, "SphereCollisionChecker")
        .def(py::init<>(),
             "创建球体碰撞检测器实例")
        
        // 配置方法
        .def("set_obstacles", &SphereCollisionChecker::set_obstacles,
             "设置障碍物列表\n\n"
             "Args:\n"
             "    obstacles (List[AABB]): AABB 障碍物列表",
             py::arg("obstacles"))
        
        .def("set_adjacent_pairs", &SphereCollisionChecker::set_adjacent_pairs,
             "设置需要忽略碰撞的相邻球体对\n\n"
             "Args:\n"
             "    pairs (List[Tuple[int, int]]): 球体索引对列表",
             py::arg("pairs"))
        
        .def("clear_obstacles", &SphereCollisionChecker::clear_obstacles,
             "清空所有障碍物")
        
        // 碰撞检测方法
        .def("check_collisions", &SphereCollisionChecker::check_collisions,
             "检查单个关节配置的球体碰撞\n\n"
             "Args:\n"
             "    sphere_coords (List[List[float]]): 球体坐标列表，每个为 [x, y, z, r]\n\n"
             "Returns:\n"
             "    Tuple[bool, List[int], List[int]]: (是否碰撞, 碰撞标志数组, 每个球体的周期数)\n"
             "        - 第一个元素: True 表示存在碰撞\n"
             "        - 第二个元素: 每个球体的碰撞状态 (1=无碰撞, 0=有碰撞)\n"
             "        - 第三个元素: 每个球体的周期数列表",
             py::arg("sphere_coords"))
        
        .def("check_collisions_batch", &SphereCollisionChecker::check_collisions_batch,
             "批量检查多个关节配置的球体碰撞\n\n"
             "Args:\n"
             "    batch_coords (List[List[List[float]]]): 批量球体坐标\n\n"
             "Returns:\n"
             "    List[Tuple[bool, List[int], List[int]]]: 每个配置的碰撞检测结果(碰撞状态, 标志数组, 各球体周期数)",
             py::arg("batch_coords"))
        
        // 查询方法
        .def("get_obstacle_count", &SphereCollisionChecker::get_obstacle_count,
             "获取当前障碍物数量")
        
        .def("get_adjacent_pairs_count", &SphereCollisionChecker::get_adjacent_pairs_count,
             "获取当前邻接对数量")
        
        .def("__repr__",
             [](const SphereCollisionChecker &checker) {
                 return "<SphereCollisionChecker obstacles=" + 
                        std::to_string(checker.get_obstacle_count()) + 
                        " adjacent_pairs=" + 
                        std::to_string(checker.get_adjacent_pairs_count()) + ">";
             });
}
