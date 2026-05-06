cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerVesselAnalyzer)

find_package(Slicer COMPONENTS ConfigurePrerequisites REQUIRED)
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

set(EXTENSION_HOMEPAGE "https://github.com/yourname/SlicerVesselAnalyzer")
set(EXTENSION_CATEGORY "Vascular")
set(EXTENSION_CONTRIBUTORS "Benny Zamir")
set(EXTENSION_DESCRIPTION "Vessel centerline extraction, IVUS navigation, diameter measurement, stent planning, and AI-assisted clinical reporting.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/Benny_Zamir/SlicerVesselAnalyzer/main/VesselAnalyzer/VesselAnalyzer_icon.png")
set(EXTENSION_SCREENSHOTURLS "https://...")
set(EXTENSION_DEPENDS "SlicerVMTK")  # your VMTK dependency

add_subdirectory(VesselAnalyzer)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})