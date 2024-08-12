file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${IceSYCL_SOURCE_DIR}/include/IceSYCL")

add_library(IceSYCL_library IceSYCL.cpp ${HEADER_LIST})

target_include_directories(IceSYCL_library PUBLIC ../include)

target_compile_features(IceSYCL_library PUBLIC cxx_std_17)