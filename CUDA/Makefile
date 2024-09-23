
# Variables
CXX = clang++
CXXFLAGS = -Wall -Wextra -std=c++17 -O0
LDFLAGS = # Add any libraries you want to link, e.g., -lm for math or -lpthread for pthreads
SRC = add.cpp
#$(wildcard *.cpp)  # All .cpp files in the current directory
OBJ = $(SRC:.cpp=.o)
TARGET = add

# Default rule
all: $(TARGET)

# Linking the target
$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)

# Compiling .cpp files to .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up object files and executable
clean:
	rm -f $(OBJ) $(TARGET)

run: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean

