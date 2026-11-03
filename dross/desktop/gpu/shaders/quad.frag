#version 450
layout(push_constant) uniform Push { ivec4 rect; uvec4 target; uvec4 source; } p;
layout(set=0,binding=0,std430) readonly buffer Pixels { uint pixels[]; } data;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 result;
void main() {
    uint packed = p.target.z;
    uint mode = p.target.w;
    if (mode == 1 || mode == 2) {
        uvec2 at = min(uvec2(uv * vec2(p.rect.zw)), uvec2(p.rect.zw)-1);
        packed = data.pixels[p.source.x + at.y * p.source.y + at.x];
    }
    if (mode == 3) {
        uvec2 at = uvec2(gl_FragCoord.xy);
        if (at.y % 4 == 0 && at.x % 4 == (at.y % 8 / 4 * 2)) packed = 0x555d61;
    }
    result = vec4(float((packed >> 16) & 255), float((packed >> 8) & 255), float(packed & 255),
                  mode == 2 ? float(packed >> 24) : 255.0) / 255.0;
}
