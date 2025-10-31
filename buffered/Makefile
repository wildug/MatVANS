# Compiler and flags
CC = nvcc
CFLAGS = -O2 -arch=sm_75 -std=c++20 --ptxas-options=-v
# CFLAGS = -g -G -O0 -arch=sm_75 -std=c++20 # Uncomment for debugging

# Target executables
TARGETS = baseline ans-warp-stripes ans-warp-rows

# Source and object files
GLOBAL_OBJS = src/safetensor_loader.o src/safetensors.o
BASELINE_OBJS = src/baseline.o src/baseline-kernel.o
ANS_WARP_STRIPES_OBJS = src/ans-warp-stripes.o src/ans-warp-stripes-kernel.o
ANS_WARP_ROWS_OBJS = src/ans-warp-rows.o src/ans-warp-rows-kernel.o
ALL_OBJS = $(GLOBAL_OBJS) $(BASELINE_OBJS) $(ANS_WARP_STRIPES_OBJS) $(ANS_WARP_ROWS_OBJS)

all: $(TARGETS)

baseline: $(GLOBAL_OBJS) $(BASELINE_OBJS)
ans-warp-stripes: $(GLOBAL_OBJS) $(ANS_WARP_STRIPES_OBJS)
ans-warp-rows: $(GLOBAL_OBJS) $(ANS_WARP_ROWS_OBJS)

$(TARGETS): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cc
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGETS) $(ALL_OBJS)
