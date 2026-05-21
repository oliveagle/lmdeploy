//! Simple diagnostic test to check C API initialization

use lmdeploy_server::turbomind_c::{EngineConfig, TurboMind};

fn main() {
    println!("=== LMDeploy C API Simple Diagnostic ===\n");

    let model_path = "/mnt/eaget-4tb/modelscope_models/Qwen/Qwen3-4B";

    // Test 1: Create EngineConfig
    println!("Test 1: Creating EngineConfig...");
    match EngineConfig::new() {
        Ok(mut cfg) => {
            println!("  ✓ EngineConfig created successfully");

            cfg.set_session_len(4096);
            cfg.set_data_type(lmdeploy_server::turbomind_c::TM_DataType::TM_DATATYPE_FP16);
            cfg.add_device(0);
            cfg.set_nnodes(1);
            cfg.set_node_rank(0);
            cfg.set_attn_tp_size(1);
            cfg.set_attn_cp_size(1);
            cfg.set_attn_dp_size(1);
            cfg.set_mlp_tp_size(1);

            // Test 2: Create TurboMind
            println!("\nTest 2: Creating TurboMind...");
            match TurboMind::create(model_path, &mut cfg) {
                Ok(tm) => {
                    println!("  ✓ TurboMind created successfully");

                    // Test 3: Try InitFromPath
                    println!("\nTest 3: Running InitFromPath...");
                    println!("  (This will be verbose, please wait...)");
                    println!("  ==============================================");
                    match tm.init_from_path(0, model_path, true) {
                        Ok(_) => {
                            println!("  ==============================================");
                            println!("  ✓ InitFromPath succeeded!");

                            // Cleanup happens on drop
                        }
                        Err(e) => {
                            println!("  ==============================================");
                            println!("  ✗ InitFromPath failed: {:?}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("  ✗ TurboMind create failed: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("  ✗ EngineConfig create failed: {:?}", e);
        }
    }

    println!("\n=== Test completed ===");
}
