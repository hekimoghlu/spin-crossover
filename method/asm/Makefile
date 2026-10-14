all: libparity.so
parity.o: parity.asm
	nasm -f elf64 -o $@ $<
libparity.so: parity.o
	gcc -shared -o $@ $<
clean:
	rm -f parity.o libparity.so
.PHONY: all clean
