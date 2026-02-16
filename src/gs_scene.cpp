#include "gs_scene.h"
#include "gs_memory.h"

static void free_if(float *&ptr) {
    if (ptr) {
        gs_heap_free_floats(ptr);
        ptr = nullptr;
    }
}

void gs_scene_free(GaussianScene &scene) {
    free_if(scene.pos_x);
    free_if(scene.pos_y);
    free_if(scene.pos_z);
    free_if(scene.sh_r);
    free_if(scene.sh_g);
    free_if(scene.sh_b);
    free_if(scene.sh_rest);
    free_if(scene.rot_w);
    free_if(scene.rot_x);
    free_if(scene.rot_y);
    free_if(scene.rot_z);
    free_if(scene.scale_x);
    free_if(scene.scale_y);
    free_if(scene.scale_z);
    free_if(scene.opacity);
    scene.num_gaussians = 0;
}
