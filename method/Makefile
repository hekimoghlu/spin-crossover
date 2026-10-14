# Top-level build: C++ (OpenMP) kernel, NASM routine, Rust sweep.  CUDA is not built (no nvcc here).
.PHONY: all cpp asm rust test run clean
all: cpp asm rust
cpp:
	$(MAKE) -C cpp
asm:
	$(MAKE) -C asm
rust:
	cd rust && cargo build --release --offline
test: all
	python3 -m scodesign.tests
run: all
	python3 -m scodesign all
clean:
	$(MAKE) -C cpp clean
	$(MAKE) -C asm clean
	cd rust && cargo clean
