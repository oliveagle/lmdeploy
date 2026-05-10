

#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/models/llama/unified_decoder_dflash.h"
#include "src/turbomind/models/llama/DFlashDraftModel.h"
#include "src/turbomind/core/logger.h"

namespace turbomind {

bool SetupDFlash(const std::string&     model_dir,
                 const DFlashParam&     dflash_param,
                 const ModelParam&      model_param,
                 const EngineParam&     engine_param,
                 const AttentionParam&  attn_param,
                 const MoeParam&        moe_param,
                 const Context&         ctx,
                 std::unique_ptr<UnifiedDecoder>& decoder)
{
    (void)model_dir;
    (void)attn_param;
    (void)moe_param;

    if (dflash_param.draft_model_path.empty()) {
        TM_LOG_INFO("[DFlash] No draft model path specified, DFlash disabled");
        return false;
    }

    try {
        TM_LOG_INFO("[DFlash] Setting up DFlash draft model from: %s",
                    dflash_param.draft_model_path.c_str());

        // 1. Create DFlash draft model (with configurable num_spec_tokens)
        auto dflash_model = std::make_unique<DFlashDraftModel>(model_param,
                                                               engine_param,
                                                               ctx,
                                                               dflash_param.num_spec_tokens);

        // 2. Enable DFlash on decoder
        decoder->EnableDFlash(true);

        // 3. Attach draft model to decoder
        decoder->SetDFlashDraftModel(std::move(dflash_model));

        TM_LOG_INFO("[DFlash] DFlash setup successful:");
        TM_LOG_INFO("[DFlash]   num_spec_tokens: %d", dflash_param.num_spec_tokens);
        TM_LOG_INFO("[DFlash]   draft_model_path: %s", dflash_param.draft_model_path.c_str());

        return true;

    } catch (const std::exception& e) {
        TM_LOG_WARNING("[DFlash] Failed to setup DFlash: %s", e.what());
        TM_LOG_WARNING("[DFlash] Falling back to normal decoder");
        return false;
    }
}

}  // namespace turbomind
