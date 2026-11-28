fn main() {
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::copy("memory.x", out.join("memory.x")).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");
    // The TLS trust anchor (DER). Default: ISRG Root X2. AGENT_ROOT_CA selects
    // another file, which is how the negative test proves a wrong anchor is refused.
    let ca = std::env::var("AGENT_ROOT_CA").unwrap_or_else(|_| "isrg-root-x2.der".into());
    std::fs::copy(&ca, out.join("root-ca.der")).unwrap_or_else(|e| panic!("AGENT_ROOT_CA {ca}: {e}"));
    println!("cargo:rerun-if-env-changed=AGENT_ROOT_CA");
    println!("cargo:rerun-if-changed={ca}");
    // The sub-device's private key seed (64 hex characters in a file; never printed). Without one the image
    // carries an all-zero seed and the sub-device role refuses to start.
    let seed = match std::env::var("AGENT_DEVICE_SEED_FILE") {
        Ok(f) => {
            let hex = std::fs::read_to_string(&f).unwrap_or_else(|e| panic!("AGENT_DEVICE_SEED_FILE: {e}"));
            let hex = hex.trim();
            assert_eq!(hex.len(), 64, "AGENT_DEVICE_SEED_FILE must hold 64 hex characters");
            println!("cargo:rerun-if-changed={f}");
            (0..32).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex")).collect::<Vec<u8>>()
        }
        Err(_) => vec![0u8; 32],
    };
    std::fs::write(out.join("device-seed.bin"), seed).unwrap();
    println!("cargo:rerun-if-env-changed=AGENT_DEVICE_SEED_FILE");
    for v in ["AGENT_HOST", "AGENT_PORT", "AGENT_TOKEN", "AGENT_TLS", "AGENT_EXPECT", "AGENT_ROLE", "AGENT_CREATED"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
}
