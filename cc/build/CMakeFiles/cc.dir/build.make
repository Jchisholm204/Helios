# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake-3.28.0-rc2-linux-x86_64/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.28.0-rc2-linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jacob/documents/github/Helios/cc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jacob/documents/github/Helios/cc/build

# Include any dependencies generated for this target.
include CMakeFiles/cc.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cc.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cc.dir/flags.make

CMakeFiles/cc.dir/src/main.c.o: CMakeFiles/cc.dir/flags.make
CMakeFiles/cc.dir/src/main.c.o: /home/jacob/documents/github/Helios/cc/src/main.c
CMakeFiles/cc.dir/src/main.c.o: CMakeFiles/cc.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jacob/documents/github/Helios/cc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/cc.dir/src/main.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/cc.dir/src/main.c.o -MF CMakeFiles/cc.dir/src/main.c.o.d -o CMakeFiles/cc.dir/src/main.c.o -c /home/jacob/documents/github/Helios/cc/src/main.c

CMakeFiles/cc.dir/src/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/cc.dir/src/main.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jacob/documents/github/Helios/cc/src/main.c > CMakeFiles/cc.dir/src/main.c.i

CMakeFiles/cc.dir/src/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/cc.dir/src/main.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jacob/documents/github/Helios/cc/src/main.c -o CMakeFiles/cc.dir/src/main.c.s

# Object files for target cc
cc_OBJECTS = \
"CMakeFiles/cc.dir/src/main.c.o"

# External object files for target cc
cc_EXTERNAL_OBJECTS =

cc: CMakeFiles/cc.dir/src/main.c.o
cc: CMakeFiles/cc.dir/build.make
cc: CMakeFiles/cc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/jacob/documents/github/Helios/cc/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable cc"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cc.dir/build: cc
.PHONY : CMakeFiles/cc.dir/build

CMakeFiles/cc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cc.dir/clean

CMakeFiles/cc.dir/depend:
	cd /home/jacob/documents/github/Helios/cc/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jacob/documents/github/Helios/cc /home/jacob/documents/github/Helios/cc /home/jacob/documents/github/Helios/cc/build /home/jacob/documents/github/Helios/cc/build /home/jacob/documents/github/Helios/cc/build/CMakeFiles/cc.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/cc.dir/depend

