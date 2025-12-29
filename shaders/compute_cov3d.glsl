#version 430 core

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (std430, binding = 0) readonly buffer InputBuffer {
    float input_data[]; // N x 7 floats: scale(3), rot(4)
};

layout (std430, binding = 2) writeonly buffer OutputBuffer {
    float sigmas_out[]; // N x 6 floats
};


mat3 computeCov3D(vec3 scale, vec4 q)
{
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    S[2][2] = scale.z;

    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    mat3 R = mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    mat3 M = S * R;
    mat3 Sigma = transpose(M) * M;

    return Sigma;
}

void main() {

    uint i = gl_GlobalInvocationID.x;
    uint base_idx = i * 7;
    vec3 scale = vec3(input_data[base_idx + 0],
                    input_data[base_idx + 1],
                    input_data[base_idx + 2]
                    );
    vec4 rot = vec4(input_data[base_idx + 3],
                    input_data[base_idx + 4],
                    input_data[base_idx + 5],
                    input_data[base_idx + 6]
                    );

    mat3 sigma = computeCov3D(scale, rot);

    uint out_idx = i * 6;

    sigmas_out[out_idx + 0] = sigma[0][0]; // sxx
    sigmas_out[out_idx + 1] = sigma[1][1]; // syy
    sigmas_out[out_idx + 2] = sigma[2][2]; // szz

    sigmas_out[out_idx + 3] = sigma[0][1]; // sxy
    sigmas_out[out_idx + 4] = sigma[0][2]; // sxz
    sigmas_out[out_idx + 5] = sigma[1][2]; // syz
}