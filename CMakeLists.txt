cmake_minimum_required(VERSION 3.5)
message("CMake prefix path: " ${CMAKE_PREFIX_PATH})
project(PRISM)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE RELEASE)

# set warning level
if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w -O3 -fPIC -g")
endif ( CMAKE_COMPILER_IS_GNUCC )
if ( MSVC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /W0 /utf-8 /Ox /permissive-")
endif ( MSVC )

# PRISM custom options
set(PRISMATIC_ENABLE_GUI 0 CACHE BOOL PRISMATIC_ENABLE_GUI)
set(PRISMATIC_ENABLE_GPU 0 CACHE BOOL PRISMATIC_ENABLE_GPU)
set(PRISMATIC_ENABLE_CLI 1 CACHE BOOL PRISMATIC_ENABLE_GPU)
#set(PRISMATIC_ENABLE_PYTHON_GPU 0 CACHE BOOL PRISMATIC_ENABLE_PYTHON_GPU)
set(PRISMATIC_ENABLE_DOUBLE_PRECISION 0 CACHE BOOL PRISMATIC_ENABLE_DOUBLE_PRECISION)
set(PRISMATIC_ENABLE_PYPRISMATIC 0 CACHE BOOL PRISMATIC_ENABLE_PYPRISMATIC)
set(PRISMATIC_USE_HDF5_STATIC 0 CACHE BOOL PRISMATIC_USE_HDF5_STATIC)
set(PRISMATIC_TESTS 0 CACHE BOOL PRISMATIC_TESTS)
set(OUTPUT_NAME prismatic CACHE STRING OUTPUT_NAME)

#set (CMAKE_BUILD_TYPE DEBUG)
if (PRISMATIC_ENABLE_GPU)
    set(NVCC_FLAGS_EXTRA "-arch=sm_60 -std=c++11 -Xcompiler -fPIC")
endif (PRISMATIC_ENABLE_GPU)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

if (UNIX)
	set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/lib)
	link_directories(${CMAKE_LIBRARY_PATH})
endif (UNIX)

# create variables with various bundles of source files
set(SOURCE_FILES
        src/configure.cpp
        src/WorkDispatcher.cpp
        src/Multislice_calcOutput.cpp
        src/PRISM01_calcPotential.cpp
        src/PRISM02_calcSMatrix.cpp
        src/PRISM03_calcOutput.cpp
        src/PRISM_entry.cpp
        src/Multislice_entry.cpp
        src/HRTEM_entry.cpp
        src/projectedPotential.cpp
        src/parseInput.cpp
        src/utility.cpp
        src/atom.cpp
        src/go.cpp
        src/fileIO.cpp
        src/probe.cpp
        src/aberration.cpp
        src/MS_MISER_entry.cpp
        src/MS_MISER_calcOutput.cpp)

if (PRISMATIC_ENABLE_GUI)
set(GUI_SOURCE_FILES
        Qt/main.cpp
        Qt/prismmainwindow.cpp
	    Qt/prism_qthreads.cpp
        Qt/prism_progressbar.cpp
        Qt/saveatomiccoordinatesdialog.cpp
        Qt/prism_colormapper.cpp
	    Qt/prism_resources.qrc
        Qt/breeze.qrc)
endif (PRISMATIC_ENABLE_GUI)

if (PRISMATIC_ENABLE_GPU)
    set(CUDA_SOURCE_FILES
            src/Multislice_calcOutput.cu
            src/PRISM02_calcSMatrix.cu
            src/PRISM03_calcOutput.cu
            src/utility.cu
            src/fileIO.cu)
endif (PRISMATIC_ENABLE_GPU)

if (PRISMATIC_TESTS)
    set(TEST_SOURCE_FILES
            unittests/testSuite.cpp
            unittests/potentialTests.cpp
            unittests/ioTests.cpp
            unittests/hrtemTests.cpp
            unittests/processingTests.cpp
            unittests/probeTests.cpp
            unittests/aberrationsTests.cpp
            unittests/seriesTests.cpp
            unittests/refocusTests.cpp
            )
endif (PRISMATIC_TESTS)

# find core packages
find_package (Threads REQUIRED)
find_package (Boost REQUIRED)

if(PRISMATIC_ENABLE_DOUBLE_PRECISION)
	if(UNIX)
		set(FFTW_FIND_COMPONENTS "DOUBLE_LIB" "DOUBLE_THREADS_LIB")
	else(UNIX)
		set(FFTW_FIND_COMPONENTS "DOUBLE_LIB")
	endif(UNIX)
else(PRISMATIC_ENABLE_DOUBLE_PRECISION)
	if(UNIX)
		set(FFTW_FIND_COMPONENTS "FLOAT_LIB" "FLOAT_THREADS_LIB")
	else(UNIX)
		set(FFTW_FIND_COMPONENTS "FLOAT_LIB")
	endif(UNIX)
endif(PRISMATIC_ENABLE_DOUBLE_PRECISION)
find_package (FFTW REQUIRED COMPONENTS ${FFTW_FIND_COMPONENTS})

if(FFTW_FOUND)
	message("Found FFTW: ${FFTW_LIBRARIES}")
else(FFTW_FOUND)
    message("Could not find FFTW library")
endif(FFTW_FOUND)

if(PRISMATIC_ENABLE_PYPRISMATIC)
    find_package (PythonInterp 3.5 REQUIRED)
    find_package (PythonLibs 3.5 REQUIRED)
endif(PRISMATIC_ENABLE_PYPRISMATIC)

if (PRISMATIC_USE_HDF5_STATIC)
   set(HDF5_USE_STATIC_LIBRARIES ON)
endif (PRISMATIC_USE_HDF5_STATIC)

set(HDF5_FIND_COMPONENTS "C" "HL" "CXX")
find_package (HDF5 REQUIRED ${HDF5_FIND_COMPONENTS})
find_library(HDF5_C_HL_LIBRARIES hdf5_hl)
find_library(HDF5_C_LIBRARIES hdf5)
find_library(HDF5_CXX_HL_LIBRARIES hdf5_hl_cpp)
find_library(HDF5_CXX_LIBRARIES hdf5_cpp)

message("Boost dir " ${Boost_INCLUDE_DIRS})
message("FFTW dir " ${FFTW_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include
                    ${CMAKE_SOURCE_DIR}/unittests/include
                    ${Boost_INCLUDE_DIRS}
                    ${FFTW_INCLUDE_DIRS}
                    ${HDF5_INCLUDE_DIRS})
# find CUDA package, if building for GPU
if (PRISMATIC_ENABLE_GPU)
    message("GPU support enabled")
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${NVCC_FLAGS_EXTRA})
endif(PRISMATIC_ENABLE_GPU)

# find Qt modules, if building GUI
if (PRISMATIC_ENABLE_GUI)
    find_package(Qt5Widgets REQUIRED)
    message(Qt5Widgets_DIR = ${Qt5Widgets_DIR})
endif (PRISMATIC_ENABLE_GUI)

# add preprocessor directives based on configuration
if (PRISMATIC_ENABLE_DOUBLE_PRECISION)
	message("Double precision enabled")
    add_definitions(-DPRISMATIC_ENABLE_DOUBLE_PRECISION)
else (PRISMATIC_ENABLE_DOUBLE_PRECISION)
	message("Single precision enabled")
endif (PRISMATIC_ENABLE_DOUBLE_PRECISION) 

if (PRISMATIC_ENABLE_GUI)
    add_definitions(-DPRISMATIC_ENABLE_GUI)
endif (PRISMATIC_ENABLE_GUI)

if (PRISMATIC_ENABLE_GPU)
    add_definitions(-DPRISMATIC_ENABLE_GPU)
endif (PRISMATIC_ENABLE_GPU)

if (PRISMATIC_ENABLE_CLI)
    add_definitions(-DPRISMATIC_ENABLE_CLI)
endif (PRISMATIC_ENABLE_CLI)


if (PRISMATIC_ENABLE_CLI)
    # build CLI
    if (PRISMATIC_ENABLE_GPU)

        cuda_add_executable(prismatic
                	     src/driver.cpp
                	     ${SOURCE_FILES}
                	     ${CUDA_SOURCE_FILES})
        cuda_add_cufft_to_target(prismatic)
    else(PRISMATIC_ENABLE_GPU)

        add_executable(prismatic
                        src/driver.cpp
                        ${SOURCE_FILES})
    endif (PRISMATIC_ENABLE_GPU)

    target_link_libraries(prismatic
        		   ${CMAKE_THREAD_LIBS_INIT}
#        		   ${Boost_LIBRARY_DIRS}
                   ${FFTW_LIBRARIES}
                   ${HDF5_LIBRARIES})
    set_target_properties(prismatic PROPERTIES OUTPUT_NAME ${OUTPUT_NAME})
endif (PRISMATIC_ENABLE_CLI)

if(APPLE)
  list(APPEND GUI_SOURCE_FILES ../Qt/icons/prismatic-icon.icns)
  set(MACOSX_BUNDLE_ICON_FILE prismatic-icon.icns)
  set(MACOSX_BUNDLE_BUNDLE_VERSION 0.1)
  set_source_files_properties(../Qt/icons/prismatic-icon.icns PROPERTIES
    MACOSX_PACKAGE_LOCATION Resources)
elseif(WIN32)
  list(APPEND GUI_SOURCE_FILES ../Qt/icons/prismatic.ico)
endif()

if (PRISMATIC_ENABLE_GUI)
    # build GUI
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR on)
    include_directories(${CMAKE_SOURCE_DIR}/Qt ${CMAKE_SOURCE_DIR}/include)
    include_directories(${Qt5Widgets_INCLUDE_DIRS})

    if (PRISMATIC_ENABLE_GPU)
        # build the shared CUDA library used by the GUI
	#cuda_add_library(prism_shared
	#                    ${SOURCE_FILES}
	#                    ${CUDA_SOURCE_FILES}
	#                    #${GUI_SOURCE_FILES}
	#                    SHARED)
	#target_link_libraries(prism_shared
	#                        ${CMAKE_THREAD_LIBS_INIT}
	#                        ${Boost_LIBRARY_DIRS}
	#                        ${FFTW_LIBRARIES})
	#cuda_add_cufft_to_target(prism_shared)
	#            add_definitions(-DPRISMATIC_BUILDING_GUI)
#        add_definitions(-DHEY)
#
        cuda_add_executable(prismatic-gui
                                ${SOURCE_FILES}
                                ${GUI_SOURCE_FILES}
                                ${CUDA_SOURCE_FILES}
				OPTIONS -DPRISMATIC_BUILDING_GUI)
        target_link_libraries(prismatic-gui
                                Qt5::Widgets
                                ${CMAKE_THREAD_LIBS_INIT}
#                                ${Boost_LIBRARY_DIRS}
                                ${FFTW_LIBRARIES}
                                ${HDF5_LIBRARIES})
        cuda_add_cufft_to_target(prismatic-gui)
    else (PRISMATIC_ENABLE_GPU)
        add_executable(prismatic-gui
                        WIN32 MACOSX_BUNDLE
                        ${SOURCE_FILES}
                        ${GUI_SOURCE_FILES})
        target_link_libraries(prismatic-gui
                                Qt5::Widgets
                                ${CMAKE_THREAD_LIBS_INIT}
#                                ${Boost_LIBRARY_DIRS}
                                ${FFTW_LIBRARIES}
                                ${HDF5_LIBRARIES})
    endif (PRISMATIC_ENABLE_GPU)
    target_compile_definitions(prismatic-gui PRIVATE PRISMATIC_BUILDING_GUI=1)

    if(APPLE)
        install(TARGETS prismatic-gui DESTINATION Applications COMPONENT runtime)
    else(APPLE)
        install(TARGETS prismatic-gui RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
    endif(APPLE)
endif (PRISMATIC_ENABLE_GUI)

if (PRISMATIC_ENABLE_CLI)
    install(TARGETS prismatic RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif(PRISMATIC_ENABLE_CLI)

# if (PRISMATIC_ENABLE_PYTHON_GPU AND PRISMATIC_ENABLE_GPU)
#     cuda_add_library(cuprismatic SHARED
# 	    	         ${SOURCE_FILES}
#                      ${CUDA_SOURCE_FILES})
#     cuda_add_cufft_to_target(cuprismatic)
#     target_link_libraries(cuprismatic
#                           ${FFTW_LIBRARIES}
# 			  ${HDF5_LIBRARIES})
#     target_compile_definitions(cuprismatic PRIVATE PRISMATIC_ENABLE_GPU=1 BUILDING_CUPRISMATIC=1 CUPRISMATIC_EXPORT=1)
#     install(TARGETS cuprismatic
# 	        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
# 	        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
# endif(PRISMATIC_ENABLE_PYTHON_GPU AND PRISMATIC_ENABLE_GPU)

if(PRISMATIC_ENABLE_PYPRISMATIC)

    include_directories(${PYTHON_INCLUDE_DIRS})
    if (PRISMATIC_ENABLE_GPU)
        cuda_add_library(pyprismatic_core SHARED pyprismatic/core.cpp ${SOURCE_FILES} ${CUDA_SOURCE_FILES})
        cuda_add_cufft_to_target(pyprismatic_core)
    else (PRISMATIC_ENABLE_GPU)
        add_library(pyprismatic_core SHARED pyprismatic/core.cpp ${SOURCE_FILES})
    endif(PRISMATIC_ENABLE_GPU)

    target_link_libraries(pyprismatic_core
                    ${CMAKE_THREAD_LIBS_INIT}
                    ${FFTW_LIBRARIES}
                    ${HDF5_LIBRARIES})

    set_target_properties(
    pyprismatic_core
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "core"
        LINKER_LANGUAGE CXX
    )

    if (WIN32)
        target_link_libraries(pyprismatic_core ${PYTHON_LIBRARIES})
        set_target_properties(pyprismatic_core PROPERTIES SUFFIX ".pyd")
    elseif (APPLE)
        set_target_properties(pyprismatic_core PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
        set_target_properties(pyprismatic_core PROPERTIES SUFFIX ".so")
    endif ()
    # else Linux: dont link

endif(PRISMATIC_ENABLE_PYPRISMATIC)

if (PRISMATIC_TESTS)
    find_package (Boost REQUIRED COMPONENTS unit_test_framework)

    if (PRISMATIC_ENABLE_GPU)
        cuda_add_executable(prismatic-tests
                        ${SOURCE_FILES}
                        ${CUDA_SOURCE_FILES}
                        ${TEST_SOURCE_FILES})
        cuda_add_cufft_to_target(prismatic-tests)
    else(PRISMATIC_ENABLE_GPU)
        add_executable(prismatic-tests
                        ${SOURCE_FILES}
                        ${TEST_SOURCE_FILES})
    endif (PRISMATIC_ENABLE_GPU)

    target_compile_definitions(prismatic-tests PRIVATE "BOOST_TEST_DYN_LINK=1")

    target_link_libraries(prismatic-tests
        ${CMAKE_THREAD_LIBS_INIT}
        ${FFTW_LIBRARIES}
        ${HDF5_LIBRARIES}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

    # always run tests last
    if(PRISMATIC_ENABLE_PYPRISMATIC)
        add_custom_target(PrismaticTestSuite ALL ./prismatic-tests --log_level=all --run_test=NONE
                            DEPENDS prismatic prismatic-tests pyprismatic_core)
    else(PRISMATIC_ENABLE_PYPRISMATIC)
        add_custom_target(PrismaticTestSuite ALL ./prismatic-tests --log_level=all --run_test=NONE
                            DEPENDS prismatic prismatic-tests)
    endif(PRISMATIC_ENABLE_PYPRISMATIC)


endif (PRISMATIC_TESTS)

set_target_properties(prismatic PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "\$ORIGIN:${INSTALL_RPATH}")
