

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace turbomind {

// Forward declarations
struct ModelParam;
struct EngineParam;
struct AttentionParam;
struct MoeParam;
struct Context;
template <typename T>
class UniquePtr;
class UnifiedDecoder;

/**
 * DFlash configuration parameters
 */
struct DFlashParam {
    std::string draft_model_path;
    int num_spec_tokens = 16;  // lucebox-hub/dflash uses 16 for better performance
    std::vector<int> aux_layer_ids = {1, 10, 19, 28, 37};  // Will be optimized to {1,16,31,46,61}
};

/**
 * Set up DFlash draft model on the given decoder.
 *
 * @param model_dir       Path to the model directory
 * @param dflash_param    DFlash configuration
 * @param model_param     Model parameters
 * @param engine_param    Engine parameters
 * @param attn_param      Attention parameters
 * @param moe_param       MoE parameters
 * @param ctx             Context
 * @param decoder         Existing decoder to enhance with DFlash
 * @return true on success, false on failure (falls back to normal decoder)
 */
bool SetupDFlash(const std::string&     model_dir,
                 const DFlashParam&     dflash_param,
                 const ModelParam&      model_param,
                 const EngineParam&     engine_param,
                 const AttentionParam&  attn_param,
                 const MoeParam&        moe_param,
                 const Context&         ctx,
                 std::unique_ptr<UnifiedDecoder>& decoder);

}  // namespace turbomind
