// PLY file inspector: Loads a 3DGS PLY and prints statistics
#include "gs_memory.h"
#include "gs_ply_loader.h"
#include "gs_scene.h"
#include <cstdio>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <ply_file>\n", argv[0]);
        return 1;
    }

    // Init system for CMM allocation
    if (!gs_sys_init()) {
        fprintf(stderr, "Failed to init AX system\n");
        return 1;
    }

    gs_cmm_print_status();

    GaussianScene scene;
    printf("Loading %s...\n", argv[1]);
    if (!gs_ply_load(argv[1], scene)) {
        fprintf(stderr, "Failed to load PLY: %s\n", argv[1]);
        gs_sys_deinit();
        return 1;
    }

    gs_scene_print_info(scene);
    gs_cmm_print_status();

    gs_scene_free(scene);
    gs_sys_deinit();

    return 0;
}
