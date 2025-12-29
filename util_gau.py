import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import ctypes
import util

@dataclass
class GaussianData:
    xyz: np.ndarray
    sigma: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray

    def __len__(self):
        return len(self.xyz)

    @property
    def pos_buffer(self):
        # PosBuffer (N x 3)
        return np.ascontiguousarray(self.xyz.reshape(-1))

    @property
    def sigma_buffer(self):
        # SigmaBuffer (N x 6)
        return np.ascontiguousarray(self.sigma.reshape(-1))

    @property
    def opacity_buffer(self):
        # OpacityBuffer (N x 1) → 4つずつまとめてuint32
        return self.pack_opacity_uint8_to_uint32(self.opacity.reshape(-1))

    @property
    def sh_buffer(self):
        # SHBuffer (N x sh_dim)
        return np.ascontiguousarray(self.sh.reshape(-1))

    @property
    def sh_dim(self):
        return self.sh.shape[-1]

    def pack_opacity_uint8_to_uint32(self, opacity_uint8: np.ndarray) -> np.ndarray:
        # 長さを4の倍数にパディング
        pad = (-len(opacity_uint8)) % 4
        if pad:
            opacity_uint8 = np.pad(opacity_uint8, (0, pad), 'constant')
        return opacity_uint8.view(np.uint32)

def naive_gaussian():
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)
    gau_scale = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)
    gau_sigma = compute_cov3d_in_gpu(gau_scale, gau_rot)
    gau_c = np.array([
        1, 0, 1, 
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1, 
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = (gau_c - 0.5) / 0.28209
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)
    gau_a = (gau_a * 255).clip(0, 255).astype(np.uint8)
    return GaussianData(
        gau_xyz,
        gau_sigma,
        gau_a,
        gau_c
    )

def compute_cov3d_in_gpu(scales: np.ndarray, rots: np.ndarray) -> int:
    input_data = np.concatenate((scales, rots), axis=1).astype(np.float32)
    # 入力 ssbo (binding point 0)
    input_buffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, input_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, input_data.nbytes, input_data, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input_buffer)
    # 出力 ssbo (binding point 1)
    output_size = scales.shape[0] * 6 * 4  # N x 6 x sizeof(float)
    output_buffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, output_size, None, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, output_buffer)

    num_gaussians = scales.shape[0]
    # ワークグループサイズを定義 (256 threds per workgroup)
    LOCAL_SIZE = 256
    num_groups = (num_gaussians + LOCAL_SIZE - 1) // LOCAL_SIZE
    compute_shader_program = util.load_compute_shader("shaders/cov3d.comp")
    glUseProgram(compute_shader_program)
    glDispatchCompute(num_groups, 1, 1) # 計算を開始
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT) # 計算の完了を待つ

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer)

    # glMapBufferRange を使用してデータを読み取り
    data_ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, output_size, GL_MAP_READ_BIT)

    c_pointer = ctypes.c_void_p(data_ptr)
    num_elements = num_gaussians * 6
    c_array = ctypes.cast(c_pointer, ctypes.POINTER(ctypes.c_float * num_elements))

    sigmas = np.frombuffer(c_array.contents, dtype=np.float32).copy()
    sigmas = sigmas.reshape((num_gaussians, 6))

    # マッピング解除
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

    # GPUリソースを解放
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    glDeleteBuffers(1, [input_buffer])
    glDeleteBuffers(1, [output_buffer])
    glDeleteProgram(compute_shader_program)

    return sigmas

def load_ply(path):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    extra_coeffs_num = len(extra_f_names)
    if extra_coeffs_num > 0:
        max_sh_degree = int(np.sqrt(extra_coeffs_num / 3 + 1) - 1)
    else:
        max_sh_degree = 0
    assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    sigmas = compute_cov3d_in_gpu(scales, rots)
    opacities = 1/(1 + np.exp(- opacities))  # sigmoid
    opacities = (opacities * 255).clip(0, 255).astype(np.uint8)
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                        features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)
    return GaussianData(xyz, sigmas, opacities, shs)

if __name__ == "__main__":
    gs = load_ply("C:\\Users\\MSI_NB\\Downloads\\viewers\\models\\train\\point_cloud\\iteration_7000\\point_cloud.ply")
    a = gs.flat()
    print(a.shape)
