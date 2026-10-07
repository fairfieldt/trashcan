// Match the operations in qwen_gdn_body.glsl's unprepared path.
// In particular, multiply the Q normalization factor by scale before applying it to Q.
void gdn_normalize(inout vec2 q,inout vec2 k,float eps,float scale){
    q*=inversesqrt(subgroupAdd(dot(q,q))+eps)*scale;
    k*=inversesqrt(subgroupAdd(dot(k,k))+eps);
}
vec2 gdn_gates(float alpha,float raw_beta,float dt,float A){
    float x=alpha+dt;
    float decay=exp((x>20.0?x:log(1.0+exp(x)))*A);
    float beta=1.0/(1.0+exp(-raw_beta));
    return vec2(decay,beta);
}
