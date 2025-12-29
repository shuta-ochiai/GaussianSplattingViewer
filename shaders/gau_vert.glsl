#version 430 core

#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

layout(location = 0) in vec2 position;
// layout(location = 1) in vec3 g_pos;
// layout(location = 2) in vec4 g_rot;
// layout(location = 3) in vec3 g_scale;
// layout(location = 4) in vec3 g_dc_color;
// layout(location = 5) in float g_opacity;


#define POS_IDX 0
#define SIGMA_IDX 3
#define OPACITY_IDX 9
#define SH_IDX 10

layout (std430, binding=0) buffer gaussian_data {
	float g_data[];
	// compact version of following data
	// vec3 g_pos[];
	// vec4 g_rot[];
	// vec3 g_scale[];
	// float g_opacity[];
	// vec3 g_sh[];
};
layout (std430, binding=1) buffer gaussian_order {
	int gi[];
};

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 hfovxy_focal;
uniform vec3 cam_pos;
uniform int sh_dim;
uniform float scale_modifier;
uniform int render_mod;  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 gaussian
uniform float near_plane;
uniform float far_plane;

out vec3 color;
out float alpha;
out vec3 conic;
out vec2 coordxy;  // local coordinate in quad, unit in pixel

vec3 computeCov2D(vec4 mean_view, float focal_x, float focal_y, float tan_fovx, float tan_fovy, mat3 cov3D, mat4 viewmatrix)
{
    vec4 t = mean_view;
    // why need this? Try remove this later
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    mat3 J = mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0
    );
    mat3 W = transpose(mat3(viewmatrix));
    mat3 T = W * J;

    mat3 cov = transpose(T) * transpose(cov3D) * T;
    // Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

vec3 get_vec3(int offset)
{
	return vec3(g_data[offset], g_data[offset + 1], g_data[offset + 2]);
}
vec4 get_vec4(int offset)
{
	return vec4(g_data[offset], g_data[offset + 1], g_data[offset + 2], g_data[offset + 3]);
}

void main()
{
	int boxid = gi[gl_InstanceID];
	int total_dim = 3 + 6 + 1 + sh_dim;
	int start = boxid * total_dim;
	alpha = g_data[start + OPACITY_IDX];
	// alpha culling
	if (alpha < 1.f / 255.f)
	{
		gl_Position = vec4(-100, -100, -100, 1);
		return;
	}
	vec4 g_pos = vec4(get_vec3(start + POS_IDX), 1.f);
    vec4 g_pos_view = view_matrix * g_pos;
	// near/far plane culling
	if (g_pos_view.z >= -near_plane || g_pos_view.z <= -far_plane) {
		gl_Position = vec4(-100, -100, -100, 1);
		return;
		}
    vec4 g_pos_screen = projection_matrix * g_pos_view;
	g_pos_screen.xyz = g_pos_screen.xyz / g_pos_screen.w;
    g_pos_screen.w = 1.f;
	// early culling
	if (any(greaterThan(abs(g_pos_screen.xyz), vec3(1.3))))
	{
		gl_Position = vec4(-100, -100, -100, 1);
		return;
	}

	vec3 g_sigma_diag = get_vec3(start + SIGMA_IDX);
	vec3 g_sigma_off_diag = get_vec3(start + SIGMA_IDX + 3);
    mat3 cov3d = mat3(
		g_sigma_diag.x, g_sigma_off_diag.x, g_sigma_off_diag.y,
		g_sigma_off_diag.x, g_sigma_diag.y, g_sigma_off_diag.z,
		g_sigma_off_diag.y, g_sigma_off_diag.z, g_sigma_diag.z
	) * scale_modifier * scale_modifier;
    vec2 wh = 2 * hfovxy_focal.xy * hfovxy_focal.z;
    vec3 cov2d = computeCov2D(g_pos_view, 
                              hfovxy_focal.z, 
                              hfovxy_focal.z, 
                              hfovxy_focal.x, 
                              hfovxy_focal.y, 
                              cov3d, 
                              view_matrix);
	// 小さいもの（直径 < 1px）は描画しない（カメラ負荷軽減）
	// cov2d.x / cov2d.z はそれぞれスクリーン空間での分散（単位: px^2）
	float cov2d_x = sqrt(max(cov2d.x, 0.0));
	float cov2d_y = sqrt(max(cov2d.z, 0.0));
	// 直径 = 2 * sqrt(cov2d). 直径 < 1px => cov2d < 0.25px^2
	if (max(cov2d_x, cov2d_y) < 0.25) {
		gl_Position = vec4(-100, -100, -100, 1);
		return;
		}

    // Invert covariance (EWA algorithm)
	float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
	if (det == 0.0f)
		gl_Position = vec4(0.f, 0.f, 0.f, 0.f);
    
    float det_inv = 1.f / det;
	conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    
    vec2 quadwh_scr = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));  // screen space half quad height and width
    vec2 quadwh_ndc = quadwh_scr / wh * 2;  // in ndc space
    g_pos_screen.xy = g_pos_screen.xy + position * quadwh_ndc;
    coordxy = position * quadwh_scr;
    gl_Position = g_pos_screen;
    


	if (render_mod == -1)
	{
		float depth = -g_pos_view.z;
		depth = depth < 0.05 ? 1 : depth;
		depth = 1 / depth;
		color = vec3(depth, depth, depth);
		return;
	}

	// Covert SH to color
	int sh_start = start + SH_IDX;
	vec3 dir = g_pos.xyz - cam_pos;
    dir = normalize(dir);
	color = SH_C0 * get_vec3(sh_start);
	
	if (sh_dim > 3 && render_mod >= 1)  // 1 * 3
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		color = color - SH_C1 * y * get_vec3(sh_start + 1 * 3) + SH_C1 * z * get_vec3(sh_start + 2 * 3) - SH_C1 * x * get_vec3(sh_start + 3 * 3);

		if (sh_dim > 12 && render_mod >= 2)  // (1 + 3) * 3
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			color = color +
				SH_C2_0 * xy * get_vec3(sh_start + 4 * 3) +
				SH_C2_1 * yz * get_vec3(sh_start + 5 * 3) +
				SH_C2_2 * (2.0f * zz - xx - yy) * get_vec3(sh_start + 6 * 3) +
				SH_C2_3 * xz * get_vec3(sh_start + 7 * 3) +
				SH_C2_4 * (xx - yy) * get_vec3(sh_start + 8 * 3);

			if (sh_dim > 27 && render_mod >= 3)  // (1 + 3 + 5) * 3
			{
				color = color +
					SH_C3_0 * y * (3.0f * xx - yy) * get_vec3(sh_start + 9 * 3) +
					SH_C3_1 * xy * z * get_vec3(sh_start + 10 * 3) +
					SH_C3_2 * y * (4.0f * zz - xx - yy) * get_vec3(sh_start + 11 * 3) +
					SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * get_vec3(sh_start + 12 * 3) +
					SH_C3_4 * x * (4.0f * zz - xx - yy) * get_vec3(sh_start + 13 * 3) +
					SH_C3_5 * z * (xx - yy) * get_vec3(sh_start + 14 * 3) +
					SH_C3_6 * x * (xx - 3.0f * yy) * get_vec3(sh_start + 15 * 3);
			}
		}
	}
	color += 0.5f;
}
