本包是给nokov动捕系统使用的ROS包，基于vrpn魔改

使用前需要执行

`sudo apt-get install ros-ROSVERSION-vrpn`

附带的ekf_pose是和px4controller包里的是一样的，接收odometry类型数据

### 对应PX4-Controller中5月17日的更新

* vrpn中增加了本机时间和动捕主机时间的时间差同步
* ekf中增加了时间同步与异常值检查
* 如果编译报错Could NOT find VRPN (missing CMAKE_HAVE_THREADS_LIBRARY) roscd vrpn;修改VRPNConfig.cmake的115行，注释掉再编译
  ```sh
       if(NOT WIN32)
        find_package(Threads ${_vrpn_quiet})
        list(APPEND _deps_libs ${CMAKE_THREAD_LIBS_INIT})
        #list(APPEND _deps_check CMAKE_HAVE_THREADS_LIBRARY)
       endif()
  ```
