//! Direct test: call InitFromPath via FFI but with file-based logging

use std::fs::OpenOptions;
use std::io::Write;

use lmdeploy_server::turbomind_c::{EngineConfig, TurboMind};

fn main() {
    let mut log = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("/tmp/diag.log")
        .unwrap();

    let model_path = "/mnt/eaget-4tb/modelscope_models/Qwen/Qwen3-4B";

    // Step 1
    writeln!(log, "STEP 1: Creating EngineConfig...").unwrap();
    let mut cfg = EngineConfig::new().expect("EngineConfig::new failed");
    cfg.set_session_len(4096);
    cfg.set_data_type(lmdeploy_server::turbomind_c::TM_DataType::TM_DATATYPE_FP16);
    cfg.add_device(0);
    cfg.set_nnodes(1);
    cfg.set_node_rank(0);
    cfg.set_attn_tp_size(1);
    cfg.set_attn_cp_size(1);
    cfg.set_attn_dp_size(1);
    cfg.set_mlp_tp_size(1);
    writeln!(log, "STEP 1: EngineConfig created").unwrap();

    // Step 2
    writeln!(log, "STEP 2: Creating TurboMind...").unwrap();
    let tm = TurboMind::create(model_path, &mut cfg).expect("TurboMind::create failed");
    writeln!(log, "STEP 2: TurboMind created").unwrap();

    // Step 3 - this is where it hangs
    writeln!(log, "STEP 3: About to call init_from_path...").unwrap();
    log.flush().unwrap();

    match tm.init_from_path(0, model_path, true) {
        Ok(_) => writeln!(log, "STEP 3: init_from_path SUCCESS").unwrap(),
        Err(e) => writeln!(log, "STEP 3: init_from_path FAILED: {:?}", e).unwrap(),
    }

    log.flush().unwrap();
    writeln!(log, "DONE").unwrap();
}
