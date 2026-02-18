use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn make_temp_dir(prefix: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "te-rs-integration-{prefix}-{}-{timestamp}",
        std::process::id()
    ));
    fs::create_dir_all(&directory).expect("create temp directory");
    directory
}

fn te_rs_binary_path() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_te-rs") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_te_rs") {
        return PathBuf::from(path);
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target/debug/te-rs")
}

#[test]
fn te_rs_help_shows_command_tree() {
    let output = Command::new(te_rs_binary_path())
        .arg("--help")
        .output()
        .expect("run te-rs --help");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("index"));
    assert!(stdout.contains("health"));
}

#[test]
fn te_rs_indexes_and_queries_files_end_to_end() {
    let root = make_temp_dir("index-query");
    fs::write(
        root.join("guide.rst"),
        "Guide\n=====\n\n.. note::\n   Important note.\n",
    )
    .expect("write fixture");

    let output = Command::new(te_rs_binary_path())
        .arg("index")
        .arg("files")
        .arg(root.to_string_lossy().to_string())
        .arg("--query")
        .arg("Important")
        .arg("--top-k")
        .arg("1")
        .output()
        .expect("run te-rs index files");

    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Indexed"));
    assert!(stdout.contains("Query: Important"));

    fs::remove_dir_all(root).ok();
}
