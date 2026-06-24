cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(ArashMagicBox)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/ArashMagicBox")
set(EXTENSION_CATEGORY "Nasal")
set(EXTENSION_CONTRIBUTORS "Tianze Kuang")
set(EXTENSION_DESCRIPTION "Builds model of nasal cavity")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/ArashMagicBox.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/ArashMagicBox/Screenshots/1.png")
set(EXTENSION_DEPENDS "AirwaySegmentation TotalSegmentator")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(NasalModel)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
