all: bin bin/cif2input bin/Vext_cpu
	cp -r bin ../..

bin:
	mkdir bin

bin/cif2input: MOFGAN/cif2input.c
	gcc MOFGAN/cif2input.c -o bin/cif2input


bin/Vext_cpu: MOFGAN/Vext_cpu.c
	g++ MOFGAN/Vext_cpu.c -o bin/Vext_cpu
