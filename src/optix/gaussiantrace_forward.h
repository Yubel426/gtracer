#pragma once
#include "auxiliary.h"

#include <optix.h>

namespace gtracer {

struct Gaussiantrace_forward {
	struct Params {
		const glm::vec3* ray_origins;
		const glm::vec3* ray_directions;
		const int* gs_idxs;
		const int3* faces;
		const glm::vec3* vertices;
		const float* opacity;
		const glm::mat3x3* SinvR;
		const glm::vec3* shs;
		glm::vec3* colors;
		float* depths;
		float* alpha;
		float alpha_min;
		float transmittance_min;
		int deg;
		int max_coeffs;
		OptixTraversableHandle handle;
	};

	struct RayGenData {};
	struct MissData {};
	struct HitGroupData {};
};

}
