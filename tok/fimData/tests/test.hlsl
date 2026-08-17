struct PSInput {
    float4 pos : SV_POSITION;
};

float4 add(float4 a, float4 b) {
    return a + b;
}

PSInput main(float4 pos : SV_POSITION) {
    PSInput outv;
    return outv;
}
