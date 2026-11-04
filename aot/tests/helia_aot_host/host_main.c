/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host harness for the helia-aot delegate tests: runs a heliaAOT module that
 * helia-torch generated, through its generated registration file and the
 * runtime registry (src/nsx_helia_aot_registry.c), on raw tensor files.
 *
 *   host_run <module> <num_inputs> <in0.bin> ... <num_outputs> <out0.bin>:<nbytes> ...
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nsx_helia_aot.h"

int main(int argc, char **argv)
{
    if (argc < 3) return 64;
    const nsx_helia_aot_model_t *m = nsx_helia_aot_find(argv[1]);
    if (m == NULL) { fprintf(stderr, "module %s not registered\n", argv[1]); return 65; }
    if (m->init() != 0) { fprintf(stderr, "init failed\n"); return 66; }
    int arg = 2;
    unsigned n_in = (unsigned)atoi(argv[arg++]);
    if (n_in != m->num_inputs) { fprintf(stderr, "inputs %u != %u\n", n_in, (unsigned)m->num_inputs); return 67; }
    for (unsigned i = 0; i < n_in; ++i) {
        FILE *f = fopen(argv[arg++], "rb");
        if (!f) return 68;
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (fread(m->input(i), 1, (size_t)n, f) != (size_t)n) return 69;
        fclose(f);
    }
    if (m->run() != 0) { fprintf(stderr, "run failed\n"); return 70; }
    unsigned n_out = (unsigned)atoi(argv[arg++]);
    if (n_out != m->num_outputs) return 71;
    for (unsigned o = 0; o < n_out; ++o) {
        char *spec = argv[arg++];
        char *colon = strrchr(spec, ':');
        if (!colon) return 72;
        *colon = '\0';
        size_t nbytes = (size_t)strtoul(colon + 1, NULL, 10);
        FILE *f = fopen(spec, "wb");
        if (!f) return 73;
        fwrite(m->output(o), 1, nbytes, f);
        fclose(f);
    }
    return 0;
}
