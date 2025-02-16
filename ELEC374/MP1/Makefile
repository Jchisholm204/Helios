# Jacob Chisholm

.PHONY: all build cmake clean format

BUILD_DIR := build
BUILD_OBJ := elec374_MP1
BUILD_TYPE ?= Debug

all: build

${BUILD_DIR}/Makefile:
	cmake \
		-B${BUILD_DIR} \
		-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-G "Unix Makefiles" \

cmake: ${BUILD_DIR}/Makefile

build: cmake
	$(MAKE) -C ${BUILD_DIR} --no-print-directory

SRCS := $(shell find . -name '*.[ch]' -or -name '*.[ch]pp')
%.format: %
	clang-format -i $<
format: $(addsuffix .format, ${SRCS})

clean:
	rm -rf $(BUILD_DIR)

run: build
	./${BUILD_DIR}/${BUILD_OBJ}

profile: build
	nsys nvprof ./${BUILD_DIR}/${BUILD_OBJ}
