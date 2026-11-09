/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host harness for compile_model(): calls a heliaAOT module that helia-torch
 * compiled from a whole PyTorch model directly, the way an application does --
 * no ExecuTorch runtime, PTE or delegate registry.
 *
 *   cc -DHAOT_NAME=<name> ... standalone_main.c <module sources>
 *   standalone_run <num_inputs> <in0.bin> ... <num_outputs> <out0.bin>:<nbytes> ...
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HAOT_CAT_(a, b) a##b
#define HAOT_CAT(a, b) HAOT_CAT_(a, b)
#define HAOT_STR_(a) #a
#define HAOT_STR(a) HAOT_STR_(a)

#include HAOT_STR(HAOT_CAT(HAOT_NAME, _model.h))

static HAOT_CAT(HAOT_NAME, _model_context_t) ctx;

int main(int argc, char **argv)
{
    if (argc < 2) return 64;
    if (HAOT_CAT(HAOT_NAME, _model_init)(&ctx) != 0) { fprintf(stderr, "init failed\n"); return 66; }
    int arg = 1;
    unsigned n_in = (unsigned)atoi(argv[arg++]);
    for (unsigned i = 0; i < n_in; ++i) {
        FILE *f = fopen(argv[arg++], "rb");
        if (!f) return 68;
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (fread((void *)ctx.inputs[i].data, 1, (size_t)n, f) != (size_t)n) return 69;
        fclose(f);
    }
    if (HAOT_CAT(HAOT_NAME, _model_run)(&ctx) != 0) { fprintf(stderr, "run failed\n"); return 70; }
    unsigned n_out = (unsigned)atoi(argv[arg++]);
    for (unsigned o = 0; o < n_out; ++o) {
        char *spec = argv[arg++];
        char *colon = strrchr(spec, ':');
        if (!colon) return 72;
        *colon = '\0';
        size_t nbytes = (size_t)strtoul(colon + 1, NULL, 10);
        FILE *f = fopen(spec, "wb");
        if (!f) return 73;
        fwrite((const void *)ctx.outputs[o].data, 1, nbytes, f);
        fclose(f);
    }
    return 0;
}
