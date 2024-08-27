

# Make an automatic library - will be static or dynamic based on user setting
add_library(IceSYCL_NativeAPI SHARED "${IceSYCL_SOURCE_DIR}/src/native_api/native_api.cpp" "${IceSYCL_SOURCE_DIR}/include/IceSYCL/native_api/native_api.hpp" )
#add_library(Abominable_library Abominable.cpp ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(IceSYCL_NativeAPI PUBLIC ../include)

target_link_libraries (IceSYCL_NativeAPI PUBLIC IceSYCL_library)

target_compile_features(IceSYCL_NativeAPI PUBLIC cxx_std_17)

#
#add_custom_command(TARGET IceSYCL_NativeAPI POST_BUILD
#        COMMAND "${CMAKE_COMMAND}" -E copy
#        "$<TARGET_FILE:AbominableUnityAPI>"
#        "${Abominable_SOURCE_DIR}/Frosty/Assets/$<TARGET_FILE_NAME:AbominableUnityAPI>"
#        COMMENT "Copying AbominableUnityAPI library to Unity project directory")
#
# IDEs should put the headers in a nice place
#source_group(
#        TREE "${Abominable_SOURCE_DIR}/include"
#        PREFIX "Header Files"
#        FILES ${HEADER_LIST})