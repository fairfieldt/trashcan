# Build on incite: make
SPV = spin0.spv spin1.spv spin2.spv spin3.spv

xgpu: xgpu.cpp $(SPV)
	g++ -O2 -g -std=c++17 -Wall -Wno-format-truncation -o xgpu xgpu.cpp -lvulkan -lpthread

spin%.spv: spin.comp
	glslc -O --target-env=vulkan1.2 -fshader-stage=compute -DMODE=$* spin.comp -o $@

clean:
	rm -f xgpu $(SPV)
