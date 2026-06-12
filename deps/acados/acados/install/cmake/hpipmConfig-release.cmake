#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hpipm" for configuration "Release"
set_property(TARGET hpipm APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hpipm PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhpipm.so"
  IMPORTED_SONAME_RELEASE "libhpipm.so"
  )

list(APPEND _cmake_import_check_targets hpipm )
list(APPEND _cmake_import_check_files_for_hpipm "${_IMPORT_PREFIX}/lib/libhpipm.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
