import os
import numpy as np
import torch
from gtracer import _C

class _GaussianTrace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bvh, rays_o, rays_d, gs_idxs, faces, vertices, opacity, SinvR, shs, alpha_min, transmittance_min, deg):    
        colors = torch.zeros_like(rays_o)
        depth = torch.zeros_like(rays_o[:, 0])
        alpha = torch.zeros_like(rays_o[:, 0])
        bvh.trace_forward(
            rays_o, rays_d, gs_idxs, faces, vertices, opacity, SinvR, shs, 
            colors, depth, alpha, 
            alpha_min, transmittance_min, deg,
        )
        
        # Keep relevant tensors for backward
        ctx.alpha_min = alpha_min
        ctx.transmittance_min = transmittance_min
        ctx.deg = deg
        ctx.bvh = bvh
        ctx.save_for_backward(rays_o, rays_d, gs_idxs, faces, vertices, opacity, SinvR, shs, colors, depth, alpha)
        return colors, depth, alpha

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, grad_out_alpha):
        rays_o, rays_d, gs_idxs, means3D, opacity, SinvR, shs, colors, depth, alpha = ctx.saved_tensors
        grad_means3D = torch.zeros_like(means3D)
        grad_opacity = torch.zeros_like(opacity)
        grad_SinvR = torch.zeros_like(SinvR)
        grad_shs = torch.zeros_like(shs)
        
        ctx.bvh.trace_backward(
            rays_o, rays_d, gs_idxs, means3D, opacity, SinvR, shs, 
            colors, depth, alpha, 
            grad_means3D, grad_opacity, grad_SinvR, grad_shs,
            grad_out_color, grad_out_depth, grad_out_alpha,
            ctx.alpha_min, ctx.transmittance_min, ctx.deg,
        )
        grads = (
            None,
            None,
            None,
            None,
            grad_means3D,
            grad_opacity,
            grad_SinvR,
            grad_shs,
            None,
            None,
            None,
        )

        return grads


class GaussianTracer():
    def __init__(self, transmittance_min=0.001):
        self.impl = _C.create_gaussiantracer()
        self.transmittance_min = transmittance_min
        
    def build_bvh(self, vertices_b, faces_b, gs_idxs):
        self.faces_b = faces_b
        self.vertices_b = vertices_b
        self.gs_idxs = gs_idxs.int()
        self.impl.build_bvh(vertices_b[faces_b])

    def update_bvh(self, vertices_b, faces_b, gs_idxs):
        assert (self.faces_b == faces_b).all(), "Update bvh must keep the triangle id not change~"
        self.gs_idxs = gs_idxs.int()
        self.impl.update_bvh(vertices_b[faces_b])

    def trace(self, rays_o, rays_d, opacity, SinvR, shs, alpha_min, deg=3):
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        colors, depth, alpha = _GaussianTrace.apply(self.impl, rays_o, rays_d, self.gs_idxs, self.faces_b, self.vertices_b, opacity, SinvR, shs, alpha_min, self.transmittance_min, deg)

        colors = colors.view(*prefix, 3)
        depth = depth.view(*prefix)
        alpha = alpha.view(*prefix)
        
        return colors, depth, alpha