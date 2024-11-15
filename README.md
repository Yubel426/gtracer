# 3D Gaussian Ray Tracer

An OptiX-based differentiable 3D Gaussian Ray Tracer, inspired by the work "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes" (https://gaussiantracer.github.io/).

### Install
```bash
# clone the repo
git clone xxx
cd xxx

# use cmake to build the project for ptx file (for Optix)
rm -rf ./build && mkdir build && cd build && cmake .. && make && cd ../

# Install the package
pip install .
```

### Example usage
```bash
cd example
# Interactive viewer for 3DGS format point cloud
python renderer.py -p point_cloud.ply
```

### Acknowledgement

* Credits to [Instant-NGP](https://github.com/NVlabs/instant-ngp) and [raytracing](https://github.com/NVlabs/instant-ngp).
* Credits to the original [3D Gaussian Ray Tracing](https://gaussiantracer.github.io/) paper:
```
@article{3dgrt2024,
    author = {Nicolas Moenne-Loccoz and Ashkan Mirzaei and Or Perel and Riccardo de Lutio and Janick Martinez Esturo and Gavriel State and Sanja Fidler and Nicholas Sharp and Zan Gojcic},
    title = {3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes},
    journal = {ACM Transactions on Graphics and SIGGRAPH Asia},
    year = {2024},
}
```
