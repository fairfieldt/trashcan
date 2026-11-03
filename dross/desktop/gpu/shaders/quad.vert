#version 450
layout(push_constant) uniform Push { ivec4 rect; uvec4 target; uvec4 source; } p;
layout(location=0) out vec2 uv;
void main() {
    const vec2 corners[6] = vec2[6](vec2(0,0),vec2(1,0),vec2(0,1),vec2(0,1),vec2(1,0),vec2(1,1));
    uv = corners[gl_VertexIndex];
    vec2 pixel = vec2(p.rect.xy) + uv * vec2(p.rect.zw);
    gl_Position = vec4(pixel / vec2(p.target.xy) * 2.0 - 1.0, 0, 1);
}
