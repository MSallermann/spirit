MESSAGE( STATUS ">> -------------------------------------------------------------------- <<" )
MESSAGE( STATUS ">> --------------------- Platforms.cmake ------------------------------ <<" )
######## Apple #######################################################
if( APPLE )
    set( PLATFORM_NAME "Apple" )
    if( SPIRIT_BUNDLE_APP )
        set( OS_BUNDLE MACOSX_BUNDLE )
    endif( )
######################################################################

######## UNIX ########################################################
elseif( UNIX)
    set( PLATFORM_NAME "UNIX" )
######################################################################

######## Windows #####################################################
elseif( WIN32 )
    set( PLATFORM_NAME "Win32" )
######################################################################

######## The End #####################################################
endif( )
######## Print platform info
message( STATUS ">> We are on the platform:        " ${PLATFORM_NAME} )
if( APPLE AND SPIRIT_BUNDLE_APP )
    message( STATUS ">> Going to create .app Bundle" )
endif( )
######################################################################
message( STATUS ">> --------------------- Platforms.cmake done ------------------------- <<" )
message( STATUS ">> -------------------------------------------------------------------- <<" )