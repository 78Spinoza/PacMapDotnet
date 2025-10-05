#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "uwot_wrapper" for configuration "Release"
set_property(TARGET uwot_wrapper APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(uwot_wrapper PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libuwot.so.3.5.0"
  IMPORTED_SONAME_RELEASE "libuwot.so.3"
  )

list(APPEND _IMPORT_CHECK_TARGETS uwot_wrapper )
list(APPEND _IMPORT_CHECK_FILES_FOR_uwot_wrapper "${_IMPORT_PREFIX}/lib/libuwot.so.3.5.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
