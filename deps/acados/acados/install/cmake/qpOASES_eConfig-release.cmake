#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "qpOASES_e" for configuration "Release"
set_property(TARGET qpOASES_e APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(qpOASES_e PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libqpOASES_e.so.3.1"
  IMPORTED_SONAME_RELEASE "libqpOASES_e.so.3.1"
  )

list(APPEND _cmake_import_check_targets qpOASES_e )
list(APPEND _cmake_import_check_files_for_qpOASES_e "${_IMPORT_PREFIX}/lib/libqpOASES_e.so.3.1" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
