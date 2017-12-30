NVFLAGS  := -std=c++11 -Xptxas -v

CXXFLAGS := -fopenmp

LDFLAGS  := -lm

MPILIBS  := -I/opt/intel/compilers_and_libraries_2017.3.191/linux/mpi/intel64/include -L/opt/intel/compilers_and_libraries_2017.3.191/linux/mpi/intel64/lib -lmpi

EXES     := HW4_cuda HW4_openmp HW4_mpi

alls: $(EXES)

.PHONY: debug
debug:
	$(MAKE) NVFLAGS="-std=c++11 -g -G"

clean:
	rm -f $(EXES)

HW4_cuda: HW4_cuda.cu
	nvcc $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LDFLAGS) -o $@ $?

HW4_openmp: HW4_openmp.cu
	nvcc $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LDFLAGS) -o $@ $?

HW4_mpi: HW4_mpi.cu
	nvcc $(NVFLAGS) $(MPILIBS) -Xcompiler="$(CXXFLAGS)" -o $@ $?
