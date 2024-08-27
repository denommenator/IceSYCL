add_library(IceSYCL_library INTERFACE)

target_include_directories(IceSYCL_library INTERFACE ../include)

target_link_libraries(IceSYCL_library INTERFACE oneDPL small_la_library)

target_compile_features(IceSYCL_library INTERFACE cxx_std_17)

add_subdirectory(native_api)