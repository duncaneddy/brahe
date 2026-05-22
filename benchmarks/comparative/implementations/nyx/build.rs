//! Build script that surfaces the resolved nyx-space and anise versions
//! into compile-time string constants so the JSON metadata is accurate.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=Cargo.lock");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let lock_path = manifest_dir.join("Cargo.lock");
    let lock = fs::read_to_string(&lock_path)
        .expect("Cargo.lock should exist after `cargo build`");

    let nyx_version = extract_version(&lock, "nyx-space").unwrap_or_else(|| "unknown".to_string());
    let anise_version = extract_version(&lock, "anise").unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=NYX_SPACE_VERSION={nyx_version}");
    println!("cargo:rustc-env=ANISE_VERSION={anise_version}");
}

/// Find `name = "<pkg>"` and the immediately-following `version = "..."` line.
fn extract_version(lockfile: &str, pkg: &str) -> Option<String> {
    let needle = format!("name = \"{pkg}\"");
    let idx = lockfile.find(&needle)?;
    let tail = &lockfile[idx..];
    let v_start = tail.find("version = \"")? + "version = \"".len();
    let v_end = tail[v_start..].find('"')?;
    Some(tail[v_start..v_start + v_end].to_string())
}
