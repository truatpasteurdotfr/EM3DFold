import cupy as cp
import numpy as np

_cubic_kernel_source = '''
extern "C" __global__
void cubic_interp_kernel(
    const float* mapin, 
    float* mapout,
    const float zpix, 
    const float ypix, 
    const float xpix,
    const float apix, 
    const float shiftz, 
    const float shifty, 
    const float shiftx,
    const int nz, 
    const int ny, 
    const int nx,
    const int pextz, 
    const int pexty, 
    const int pextx
)
{
    
    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    int indy = blockIdx.y * blockDim.y + threadIdx.y;
    int indz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (indx >= pextx || indy >= pexty || indz >= pextz) return;
    
    float gx = (indx * apix + shiftx) / xpix;
    float gy = (indy * apix + shifty) / ypix;
    float gz = (indz * apix + shiftz) / zpix;
    
    int intx = (int)floor(gx);
    int inty = (int)floor(gy);
    int intz = (int)floor(gz);
    
    if (intz >= 0 && intz + 1 < nz && inty >= 0 && inty + 1 < ny && intx >= 0 && intx + 1 < nx) {
        float wz[4], wy[4], wx[4];
        
        // Calculate weights for z, y, x
        for (int dim = 0; dim < 3; dim++) {
            float* w = (dim == 0) ? wz : ((dim == 1) ? wy : wx);
            float g = (dim == 0) ? gz : ((dim == 1) ? gy : gx);
            int intg = (dim == 0) ? intz : ((dim == 1) ? inty : intx);
            
            float a = -0.5f;
            float d1 = 1.0f + (g - intg);
            float d2 = d1 - 1.0f;
            float d3 = 1.0f - d2;
            float d4 = d3 + 1.0f;
            
            w[0] = a * fabs(d1*d1*d1) - 5.0f * a * d1*d1 + 8.0f * a * fabs(d1) - 4.0f * a;
            w[1] = (a + 2.0f) * fabs(d2*d2*d2) - (a + 3.0f) * d2*d2 + 1.0f;
            w[2] = (a + 2.0f) * fabs(d3*d3*d3) - (a + 3.0f) * d3*d3 + 1.0f;
            w[3] = a * fabs(d4*d4*d4) - 5.0f * a * d4*d4 + 8.0f * a * fabs(d4) - 4.0f * a;
        }
        
        float result = 0.0f;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    int z_idx = intz + i - 1;
                    int y_idx = inty + j - 1;
                    int x_idx = intx + k - 1;
                    
                    if (z_idx >= 0 && z_idx < nz && y_idx >= 0 && y_idx < ny && x_idx >= 0 && x_idx < nx) {
                        int src_idx = z_idx * ny * nx + y_idx * nx + x_idx;
                        result += wz[i] * wy[j] * wx[k] * mapin[src_idx];
                    }
                }
            }
        }
        
        int dst_idx = indz * pexty * pextx + indy * pextx + indx;

        //atomic add
        atomicAdd(&mapout[dst_idx], result); 
    }
}
'''

class Interp3d:
    def __init__(self):
        self.mapout = None
        self.pextx = None
        self.pexty = None
        self.pextz = None
        
        self._cubic_kernel = cp.RawKernel(_cubic_kernel_source, 'cubic_interp_kernel')

    def cubic(self, mapin, zpix, ypix, xpix, apix, shiftz, shifty, shiftx, nz, ny, nx):
        # `parse_map()` may transpose the grid for non-123 CRS orders, which can
        # leave `mapin` as a non-contiguous NumPy view. The raw CUDA kernel below
        # assumes a contiguous z-y-x memory layout when flattening indices, so we
        # must materialize a contiguous copy before moving to GPU.
        mapin_gpu = cp.asarray(np.ascontiguousarray(mapin), dtype=cp.float32)
        
        pextx = int(cp.floor(xpix * (nx - 1) / apix)) + 1
        pexty = int(cp.floor(ypix * (ny - 1) / apix)) + 1
        pextz = int(cp.floor(zpix * (nz - 1) / apix)) + 1
        
        self.mapout = cp.zeros((pextz, pexty, pextx), dtype=cp.float32)
        self.pextx = pextx
        self.pexty = pexty
        self.pextz = pextz
        
        self._launch_kernel(
            self._cubic_kernel, 
            mapin_gpu, 
            self.mapout, 
            zpix, 
            ypix, 
            xpix, 
            apix, 
            shiftz, 
            shifty, 
            shiftx, 
            nz, 
            ny, 
            nx, 
            pextz, 
            pexty, 
            pextx,
        )
       
        cp.cuda.Stream.null.synchronize()

        self.mapout = cp.asnumpy(self.mapout).astype(np.float32) if self.mapout is not None else None

    def _launch_kernel(self, kernel, mapin, mapout, *args):
        pextz, pexty, pextx = args[-3:]
       
        # Max thread is 1024, 8 * 8 * 4 = 256 < 1024
        block_size = (8, 8, 4)

        grid_size = (
            (pextx + block_size[0] - 1) // block_size[0],
            (pexty + block_size[1] - 1) // block_size[1],
            (pextz + block_size[2] - 1) // block_size[2]
        )
       
        kernel(
            grid_size, 
            block_size, 
            (mapin, mapout) + tuple(cp.float32(arg) if isinstance(arg, float) else arg for arg in args))

    def del_mapout(self):
        self.mapout = None


