#include "gaussiantrace_backward.h"
#include <optix.h>


namespace gtracer {

extern "C" {
	__constant__ Gaussiantrace_backward::Params params;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	float O_final = params.alpha[idx.x];
	if (O_final==0.0f) return;

	glm::vec3 ray_o = params.ray_origins[idx.x];
	glm::vec3 ray_d = params.ray_directions[idx.x];
	glm::vec3 ray_origin;
	glm::vec3 C = glm::vec3(0.0f, 0.0f, 0.0f), C_final = params.colors[idx.x], grad_colors = params.grad_colors[idx.x];
	float grad_alpha = params.grad_alpha[idx.x];

	float T = 1.0f, t_start = 0.0f, t_curr = 0.0f;

	HitInfo hitArray[MAX_BUFFER_SIZE];
	unsigned int hitArrayPtr0 = (unsigned int)((uintptr_t)(&hitArray) & 0xFFFFFFFF);
    unsigned int hitArrayPtr1 = (unsigned int)(((uintptr_t)(&hitArray) >> 32) & 0xFFFFFFFF);

	int k=0;
	while ((t_start < T_SCENE_MAX) && (T > params.transmittance_min)){
		k++;
		ray_origin = ray_o + t_start * ray_d;
		
		for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
			hitArray[i].t = 1e16f;
			hitArray[i].primIdx = -1;
		}
		optixTrace(
			params.handle,
			make_float3(ray_origin.x, ray_origin.y, ray_origin.z),
			make_float3(ray_d.x, ray_d.y, ray_d.z),
			0.0f,                // Min intersection distance
			T_SCENE_MAX,               // Max intersection distance
			0.0f,                // rayTime -- used for motion blur
			OptixVisibilityMask(255), // Specify always visible
			OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
			0,                   // SBT offset
			1,                   // SBT stride
			0,                   // missSBTIndex
			hitArrayPtr0,
			hitArrayPtr1
		);

		for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
			int primIdx = hitArray[i].primIdx;

			if (primIdx == -1) {
				t_curr = T_SCENE_MAX;
				break;
			}
			else{
				t_curr = hitArray[i].t;
				float o = params.opacity[primIdx];
				int3 face = params.faces[primIdx];
				glm::vec3 vertex0 = params.vertices[face.x];
				glm::vec3 vertex1 = params.vertices[face.y];
				glm::vec3 vertex2 = params.vertices[face.z];

				glm::vec3 triangle_center = (vertex0 + vertex1 + vertex2) / 3.0f;
				float3 world_p = make_float3(
					ray_o.x + t_curr * ray_d.x,
					ray_o.y + t_curr * ray_d.y,
					ray_o.z + t_curr * ray_d.z
				);
				glm::vec3 relative_p = glm::vec3(world_p.x, world_p.y, world_p.z) - triangle_center;
				float G = __expf(-0.5f * glm::dot(relative_p, relative_p));
				float alpha = min(0.99f, o * G);
				if (alpha<params.alpha_min) continue;
				glm::vec3 c = computeColorFromSH_forward(params.deg, ray_d, params.shs + primIdx * params.max_coeffs);
				float w = T * alpha;
				C += w * c;

				T *= (1 - alpha);
				glm::vec3 dL_dc = grad_colors * w;
				float dL_dalpha = (
					glm::dot(grad_colors, T * c - (C_final - C)) +
					grad_alpha * (1 - O_final)
				) / max(1e-6f, 1 - alpha);
				computeColorFromSH_backward(params.deg, ray_d, params.shs + primIdx * params.max_coeffs, dL_dc, params.grad_shs + primIdx * params.max_coeffs);
				float dL_do = dL_dalpha * G;
				glm::vec3 dL_dp = -dL_dalpha * o * G * relative_p;

				glm::vec3 dL_dvertex = dL_dp / 3.0f;
				atomicAdd(&params.grad_vertices[face.x].x, dL_dvertex.x);
				atomicAdd(&params.grad_vertices[face.x].y, dL_dvertex.y);
				atomicAdd(&params.grad_vertices[face.x].z, dL_dvertex.z);

				atomicAdd(&params.grad_vertices[face.y].x, dL_dvertex.x);
				atomicAdd(&params.grad_vertices[face.y].y, dL_dvertex.y);
				atomicAdd(&params.grad_vertices[face.y].z, dL_dvertex.z);

				atomicAdd(&params.grad_vertices[face.z].x, dL_dvertex.x);
				atomicAdd(&params.grad_vertices[face.z].y, dL_dvertex.y);
				atomicAdd(&params.grad_vertices[face.z].z, dL_dvertex.z);
				
				atomicAdd(params.grad_opacity+primIdx, dL_do);
				if (T < params.transmittance_min){
					break;
				}
			}
		}
		if (t_curr==0.0f) break;
		t_start += t_curr;
	}
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __closesthit__ch() {
}

extern "C" __global__ void __anyhit__ah() {
	unsigned int hitArrayPtr0 = optixGetPayload_0();
    unsigned int hitArrayPtr1 = optixGetPayload_1();

    HitInfo* hitArray = (HitInfo*)((uintptr_t)hitArrayPtr0 | ((uintptr_t)hitArrayPtr1 << 32));

	float THit = optixGetRayTmax();
    int i_prim = optixGetPrimitiveIndex();
	HitInfo newHit = {THit, i_prim};

	for (int i = 0; i < MAX_BUFFER_SIZE; ++i) {
        if (hitArray[i].t > newHit.t) {
			host_device_swap<HitInfo>(hitArray[i], newHit);
        }
    }
	if (THit < hitArray[MAX_BUFFER_SIZE - 1].t) {
        optixIgnoreIntersection(); 
    }
}

}
