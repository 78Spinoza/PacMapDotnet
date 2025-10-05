#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pacmap_wrapper" for configuration "Debug"
set_property(TARGET pacmap_wrapper APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(pacmap_wrapper PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/pacmap.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/pacmap.dll"
  )

list(APPEND _cmake_import_check_targets pacmap_wrapper )
list(APPEND _cmake_import_check_files_for_pacmap_wrapper "${_IMPORT_PREFIX}/lib/pacmap.lib" "${_IMPORT_PREFIX}/bin/pacmap.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
