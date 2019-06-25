MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- UI - CPP-nanoGUI ----------------------------- <<" )

######### CMake Version ##########
cmake_minimum_required(VERSION 3.1)
project(Spirit_UI_CPP_nanoGUI)

# Set Output dir
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

set( CMAKE_INCLUDE_CURRENT_DIR ON )

set(SPIRIT_GUI_INCLUDES "${PROJECT_SOURCE_DIR}/include")
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
add_executable(spirit_nanoGUI main.cpp ${SPIRIT_GUI_SOURCES})

######### Add nanoGUI #########
# Disable building extras we won't need (pure C++ project)
set(NANOGUI_BUILD_EXAMPLE OFF CACHE BOOL " " FORCE)
set(NANOGUI_BUILD_PYTHON  OFF CACHE BOOL " " FORCE)
set(NANOGUI_INSTALL       OFF CACHE BOOL " " FORCE)

# Add the configurations from nanogui
add_subdirectory(thirdparty/nanogui)
# For reliability of parallel build, make the NanoGUI targets dependencies
set_property(TARGET glfw glfw_objects nanogui PROPERTY FOLDER "dependencies")
add_definitions(${NANOGUI_EXTRA_DEFS})
include_directories(${NANOGUI_EXTRA_INCS})

# Specify GLM Include DIRS
set( GLM_INCLUDE_DIRS "${CMAKE_ROOT_DIR}")

# Link libraries and include directories
target_include_directories(spirit_nanoGUI PUBLIC 
                                                    "${PROJECT_SOURCE_DIR}/thirdparty/nanogui/include" 
                                                    ${SPIRIT_GUI_INCLUDES} 
                                                    ${SPIRIT_INCLUDE_DIRS} 
                                                    ${GLAD_INCLUDE_DIRS}
                                                    ${GLM_INCLUDE_DIRS}
                                                    ${GL_INCLUDE_DIRS}
                                                    ${CMAKE_SOURCE_DIR}/VFRendering/include
                                                    )

target_link_libraries(     spirit_nanoGUI nanogui  
                                                    VFRendering
                                                    ${NANOGUI_EXTRA_LIBS} 
                                                    ${SPIRIT_LIBRARIES_STATIC} )

MESSAGE( STATUS ">> --------------------- UI - CPP-nanoGUI done ------------------------ <<" )
MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )

