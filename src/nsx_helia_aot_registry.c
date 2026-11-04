/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fixed-size registry of heliaAOT modules for HeliaAotBackend (see
 * includes-api/nsx_helia_aot.h). Filled from constructors before main(), read
 * at Method load; no locking needed on a single-threaded bare-metal target.
 */

#include "nsx_helia_aot.h"

#include <stddef.h>
#include <string.h>

static const nsx_helia_aot_model_t *s_models[NSX_HELIA_AOT_MAX_MODELS];
static unsigned s_count;

int nsx_helia_aot_register(const nsx_helia_aot_model_t *model)
{
    if (model == NULL || model->name == NULL || model->name[0] == '\0') {
        return -2;
    }
    for (unsigned i = 0; i < s_count; ++i) {
        if (strcmp(s_models[i]->name, model->name) == 0) {
            s_models[i] = model;
            return 0;
        }
    }
    if (s_count >= NSX_HELIA_AOT_MAX_MODELS) {
        return -1;
    }
    s_models[s_count++] = model;
    return 0;
}

const nsx_helia_aot_model_t *nsx_helia_aot_find(const char *name)
{
    if (name == NULL) {
        return NULL;
    }
    for (unsigned i = 0; i < s_count; ++i) {
        if (strcmp(s_models[i]->name, name) == 0) {
            return s_models[i];
        }
    }
    return NULL;
}
