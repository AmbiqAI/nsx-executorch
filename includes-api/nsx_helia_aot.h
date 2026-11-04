/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Registry between heliaAOT-generated model modules and nsx-executorch's
 * HeliaAotBackend (NSX_EXECUTORCH_ENABLE_HELIA_AOT=ON).
 *
 * helia-torch's helia-aot delegate writes one NSX module per delegated
 * partition. Each carries a small generated <name>_et_registration.c that
 * calls nsx_helia_aot_register() from a constructor; the PTE's delegate blob
 * names the module, and the backend looks it up here at Method load.
 */

#ifndef NSX_HELIA_AOT_H
#define NSX_HELIA_AOT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NSX_HELIA_AOT_MAX_MODELS
#define NSX_HELIA_AOT_MAX_MODELS 8
#endif

typedef struct nsx_helia_aot_model {
    /* Module name, as recorded in the delegate blob. */
    const char *name;
    uint32_t num_inputs;
    uint32_t num_outputs;
    /* <name>_model_init / <name>_model_run on the module's context; 0 on success. */
    int32_t (*init)(void);
    int32_t (*run)(void);
    /* Data of model input / output `index` in the module's context; NULL when out of range. */
    void *(*input)(uint32_t index);
    void *(*output)(uint32_t index);
} nsx_helia_aot_model_t;

/* 0 on success, -1 when the table is full, -2 for a NULL or unnamed model.
 * Registering the same name again replaces the earlier entry. */
int nsx_helia_aot_register(const nsx_helia_aot_model_t *model);

/* The registered model called `name`, or NULL. */
const nsx_helia_aot_model_t *nsx_helia_aot_find(const char *name);

#ifdef __cplusplus
}
#endif

#endif /* NSX_HELIA_AOT_H */
