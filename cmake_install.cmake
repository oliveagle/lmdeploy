# Install script for directory: /mnt/eaget-4tb/data/llm_server/lmdeploy

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_turbomind.cpython-312-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_turbomind.cpython-312-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_turbomind.cpython-312-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_turbomind.cpython-312-x86_64-linux-gnu.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib" TYPE MODULE FILES "/mnt/eaget-4tb/data/llm_server/lmdeploy/lib/_turbomind.cpython-312-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_turbomind.cpython-312-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_turbomind.cpython-312-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_turbomind.cpython-312-x86_64-linux-gnu.so"
         OLD_RPATH "/usr/local/cuda-12.5/targets/x86_64-linux/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_turbomind.cpython-312-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/mnt/eaget-4tb/data/llm_server/lmdeploy/src/turbomind/python/CMakeFiles/_turbomind.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_xgrammar.cpython-312-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_xgrammar.cpython-312-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_xgrammar.cpython-312-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_xgrammar.cpython-312-x86_64-linux-gnu.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib" TYPE MODULE FILES "/mnt/eaget-4tb/data/llm_server/lmdeploy/lib/_xgrammar.cpython-312-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_xgrammar.cpython-312-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_xgrammar.cpython-312-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_xgrammar.cpython-312-x86_64-linux-gnu.so"
         OLD_RPATH "/usr/local/cuda-12.5/targets/x86_64-linux/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/_xgrammar.cpython-312-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/mnt/eaget-4tb/data/llm_server/lmdeploy/src/turbomind/python/CMakeFiles/_xgrammar.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/libturbomind_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/libturbomind_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/libturbomind_c.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/libturbomind_c.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib" TYPE SHARED_LIBRARY FILES "/mnt/eaget-4tb/data/llm_server/lmdeploy/lib/libturbomind_c.so")
  if(EXISTS "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/libturbomind_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/libturbomind_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/libturbomind_c.so"
         OLD_RPATH "/usr/local/cuda-12.5/targets/x86_64-linux/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}/mnt/eaget-4tb/data/llm_server/lmdeploy/lmdeploy/lib/libturbomind_c.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/mnt/eaget-4tb/data/llm_server/lmdeploy/src/turbomind/capi/CMakeFiles/turbomind_c.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/mnt/eaget-4tb/data/llm_server/lmdeploy/include/turbomind_c.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/mnt/eaget-4tb/data/llm_server/lmdeploy/include" TYPE FILE FILES "/mnt/eaget-4tb/data/llm_server/lmdeploy/src/turbomind/capi/turbomind_c.h")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/mnt/eaget-4tb/data/llm_server/lmdeploy/_deps/yaml-cpp-build/cmake_install.cmake")
  include("/mnt/eaget-4tb/data/llm_server/lmdeploy/_deps/xgrammar-build/cmake_install.cmake")
  include("/mnt/eaget-4tb/data/llm_server/lmdeploy/src/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/mnt/eaget-4tb/data/llm_server/lmdeploy/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/mnt/eaget-4tb/data/llm_server/lmdeploy/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
