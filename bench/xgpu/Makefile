# Build on incite: make
xgpu: xgpu.cpp spin.spv
	g++ -O2 -g -std=c++17 -Wall -o xgpu xgpu.cpp -lvulkan -lpthread

spin.spv: spin.comp
	glslc -O --target-env=vulkan1.2 -fshader-stage=compute spin.comp -o spin.spv

clean:
	rm -f xgpu spin.spv
