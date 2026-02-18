//! te-core: Shared core logic for Town Elder
//!
//! This crate provides shared primitives for file scanning, git operations,
//! and document parsing. It can be compiled as:
//! - A native Rust library
//! - A Python extension module via PyO3
//! - A standalone CLI binary via clap

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

// =============================================================================
// Git Log/Commit Parsing
// =============================================================================

/// Git log field separator (ASCII 31 - Unit Separator)
const LOG_FIELD_SEPARATOR: char = '\x1f';
/// Git log record separator (ASCII 30 - Record Separator)
const LOG_RECORD_SEPARATOR: char = '\x1e';

/// Minimum parts expected in git log format output (hash, message, author, date)
const MIN_LOG_PARTS: usize = 4;
/// Number of parts in numstat output (additions, deletions, filename)
const NUMSTAT_PARTS: usize = 3;

/// Default max diff size in bytes (100KB)
pub const DEFAULT_MAX_DIFF_SIZE: usize = 100 * 1024;
/// Default file extensions included during indexing.
pub const DEFAULT_FILE_EXTENSIONS: &[&str] = &[".py", ".md", ".rst"];
/// Default directory and glob-style exclusion patterns.
pub const DEFAULT_FILE_EXCLUDES: &[&str] = &[
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "venv",
    ".env",
    ".eggs",
    "*.egg-info",
    ".hg",
    ".svn",
    ".bzr",
    "vendor",
    "_build",
];

fn should_exclude(path: &Path, exclude_patterns: &HashSet<String>) -> bool {
    for component in path.components() {
        let part = component.as_os_str().to_string_lossy();
        for pattern in exclude_patterns {
            if pattern.starts_with('*') {
                let suffix = &pattern[1..];
                if part.ends_with(suffix) {
                    return true;
                }
            } else if part == pattern.as_str() {
                return true;
            }
        }
    }
    false
}

/// Normalize absolute paths similarly to Python's `Path.resolve()` for
/// non-strict path handling used by doc-id helper parity.
fn normalize_absolute_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push(component.as_os_str());
                }
            }
            Component::Normal(part) => normalized.push(part),
        }
    }

    normalized
}

/// Scan files under a root directory with extension/exclusion parity to
/// the Python scanner implementation.
pub fn scan_files(
    root_path: &Path,
    extensions: Option<&HashSet<String>>,
    exclude_patterns: Option<&HashSet<String>>,
) -> Vec<PathBuf> {
    let extension_set: HashSet<String> = match extensions {
        Some(exts) if !exts.is_empty() => exts.clone(),
        _ => DEFAULT_FILE_EXTENSIONS
            .iter()
            .map(|ext| ext.to_string())
            .collect(),
    };

    let mut effective_excludes: HashSet<String> = DEFAULT_FILE_EXCLUDES
        .iter()
        .map(|pattern| pattern.to_string())
        .collect();
    if let Some(custom_excludes) = exclude_patterns {
        effective_excludes.extend(custom_excludes.iter().cloned());
    }

    fn collect_matching_files(
        current_dir: &Path,
        extension_set: &HashSet<String>,
        exclude_patterns: &HashSet<String>,
        files: &mut Vec<PathBuf>,
    ) {
        let entries = match std::fs::read_dir(current_dir) {
            Ok(entries) => entries,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if should_exclude(&path, exclude_patterns) {
                continue;
            }

            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };

            if file_type.is_dir() {
                collect_matching_files(&path, extension_set, exclude_patterns, files);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }

            let path_str = path.to_string_lossy();
            if extension_set.iter().any(|ext| path_str.ends_with(ext)) {
                files.push(path);
            }
        }
    }

    if should_exclude(root_path, &effective_excludes) {
        return Vec::new();
    }

    let mut files: Vec<PathBuf> = Vec::new();
    collect_matching_files(root_path, &extension_set, &effective_excludes, &mut files);

    files.sort_by_cached_key(|path| path.to_string_lossy().to_string());
    files
}

/// Represents a git commit with metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Commit {
    /// Full commit hash
    pub hash: String,
    /// Commit message (subject line)
    pub message: String,
    /// Author name
    pub author: String,
    /// Author date as ISO 8601 string
    pub date: String,
    /// List of files changed in this commit
    pub files_changed: Vec<String>,
}

/// Parse a git log commit header line.
///
/// Format: `<hash><field_separator><subject><field_separator><author><field_separator><date><record_separator>`
///
/// Returns Some(Commit) if parsing succeeds, None otherwise.
pub fn parse_commit_header(line: &str) -> Option<Commit> {
    // Remove record separator and trailing newlines
    let line = line.trim_end_matches('\n').trim_end_matches(LOG_RECORD_SEPARATOR);

    let parts: Vec<&str> = line.split(LOG_FIELD_SEPARATOR).collect();

    if parts.len() < MIN_LOG_PARTS {
        return None;
    }

    let hash = parts[0].trim().to_string();
    let message = parts[1].to_string();
    let author = parts[2].to_string();
    let date = parts[3].to_string();

    // Validate hash looks reasonable (at least 6 chars for short hash)
    if hash.len() < 6 {
        return None;
    }

    Some(Commit {
        hash,
        message,
        author,
        date,
        files_changed: Vec::new(),
    })
}

/// Parse a numstat line to extract file path.
///
/// Numstat format: `<additions><tab><deletions><tab><filename>`
/// Binary files show `-` for additions and/or deletions.
///
/// Returns Some(file_path) if parsing succeeds, None otherwise.
pub fn parse_numstat_line(line: &str) -> Option<String> {
    let line = line.trim();

    if line.is_empty() || !line.contains('\t') {
        return None;
    }

    let parts: Vec<&str> = line.split('\t').collect();

    if parts.len() < NUMSTAT_PARTS {
        return None;
    }

    let file_path = parts[2].trim();

    // Skip binary files (shown as `-`) and empty paths
    if file_path.is_empty() || file_path == "-" {
        return None;
    }

    Some(file_path.to_string())
}

/// Parse git log output with numstat into Commit objects.
///
/// Input: output from `git log --format=%H%x1f%s%x1f%an%x1f%aI%x1e --numstat`
///
/// Returns: Vector of Commit objects with files_changed populated.
pub fn parse_commits_with_numstat(output: &str) -> Vec<Commit> {
    let mut commits: Vec<Commit> = Vec::new();
    let mut current_commit: Option<Commit> = None;
    let mut current_files: Vec<String> = Vec::new();

    for line in output.lines() {
        // Check if this line starts a new commit (contains record separator)
        if line.contains(LOG_RECORD_SEPARATOR) {
            // Save previous commit if exists
            if let Some(mut commit) = current_commit.take() {
                commit.files_changed = current_files.clone();
                commits.push(commit);
            }

            // Parse new commit header
            current_commit = parse_commit_header(line);
            current_files.clear();
        } else if current_commit.is_some() {
            // Try to parse as numstat line
            if let Some(file_path) = parse_numstat_line(line) {
                current_files.push(file_path);
            }
        }
    }

    // Don't forget the last commit
    if let Some(mut commit) = current_commit {
        commit.files_changed = current_files;
        commits.push(commit);
    }

    commits
}

/// Check if a line contains the commit record separator.
pub fn is_commit_header_line(line: &str) -> bool {
    line.contains(LOG_RECORD_SEPARATOR)
}

// =============================================================================
// Diff Truncation
// =============================================================================

/// Truncate diff text if it exceeds the maximum size.
///
/// Uses byte length to determine truncation. If truncation occurs,
/// appends a message indicating the diff was truncated.
///
/// This matches the behavior in Python: `diff[:max_size] + f"\n\n[truncated - exceeded {max_size} byte limit]"`
pub fn truncate_diff(diff: &str, max_size: usize) -> String {
    if diff.len() <= max_size {
        return diff.to_string();
    }

    let truncated = diff.get(..max_size).unwrap_or(diff);
    format!("{truncated}\n\n[truncated - exceeded {max_size} byte limit]")
}

/// Check if diff text contains the truncation marker.
pub fn is_diff_truncated(diff: &str) -> bool {
    diff.contains("[truncated")
}

// =============================================================================
// Commit Text Assembly
// =============================================================================

/// Assemble commit text for embedding.
///
/// Combines the commit message with the diff text to create the final
/// text representation for embedding.
///
/// This matches the Python behavior:
/// ```python
/// text = f"Commit: {commit.message}\n\n{diff_text}"
/// ```
pub fn assemble_commit_text(message: &str, diff_text: &str) -> String {
    format!("Commit: {message}\n\n{diff_text}")
}

/// Check if the original diff was truncated and append truncation note.
///
/// If the diff contains the truncation marker, appends a note to the text.
pub fn append_truncation_note(text: &str, diff: &str) -> String {
    if is_diff_truncated(diff) {
        format!("{text} [diff was truncated due to size]")
    } else {
        text.to_string()
    }
}

// =============================================================================
// Git Blob Parsing (git ls-files --stage)
// =============================================================================

/// Valid file modes for git-tracked files
const VALID_FILE_MODES: &[&str] = &["100644", "100755"];

/// Represents a git-tracked file with its blob hash.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackedFile {
    /// Relative path from repo root
    pub path: String,
    /// 40-character SHA-1 hex digest
    pub blob_hash: String,
    /// File mode (e.g., "100644")
    pub mode: String,
}

/// Parse a single line from `git ls-files --stage` output.
///
/// Format: `<mode> <blob_hash> <stage>\t<path>`
/// Example: `100644 0acd8c61f0d9dc1e5db7ad7c2dbce2bb16b8d6de 0\t.beads/.gitignore`
///
/// Returns Some(TrackedFile) if parsing succeeds and mode is valid (100644 or 100755).
/// Returns None if:
/// - Line is empty or malformed
/// - Mode is not a valid file mode (skips symlinks, etc.)
pub fn parse_git_blob_line(line: &str) -> Option<TrackedFile> {
    if line.is_empty() {
        return None;
    }

    // Format: <mode> <blob_hash> <stage>\t<path>
    // Split on tab to preserve path (may contain spaces)
    let parts: Vec<&str> = line.split('\t').collect();
    if parts.len() != 2 {
        return None;
    }

    let metadata = parts[0];
    let file_path = parts[1];

    // Parse metadata: mode, blob_hash, stage
    let meta_parts: Vec<&str> = metadata.split_whitespace().collect();
    if meta_parts.len() != 3 {
        return None;
    }

    let mode = meta_parts[0];
    let blob_hash = meta_parts[1];
    let _stage = meta_parts[2]; // Stage is not needed for our use case

    // Skip symlinks (mode 120000) and other special files
    if !VALID_FILE_MODES.contains(&mode) {
        return None;
    }

    // Validate blob_hash is a valid SHA-1 hex (40 characters)
    if blob_hash.len() != 40 || !blob_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }

    Some(TrackedFile {
        path: file_path.to_string(),
        blob_hash: blob_hash.to_string(),
        mode: mode.to_string(),
    })
}

/// Parse git ls-files --stage output into a dictionary of TrackedFile.
///
/// Input: output from `git ls-files --stage`
/// Returns: HashMap mapping file path to TrackedFile
pub fn parse_git_blobs(output: &str) -> HashMap<String, TrackedFile> {
    let mut tracked_files: HashMap<String, TrackedFile> = HashMap::new();

    for line in output.lines() {
        if let Some(tracked_file) = parse_git_blob_line(line) {
            tracked_files.insert(tracked_file.path.clone(), tracked_file);
        }
    }

    tracked_files
}

// =============================================================================
// Git Diff Parsing
// =============================================================================

/// Git diff header prefix
const GIT_DIFF_HEADER_PREFIX: &str = "diff --git";

/// Extract the path from a git diff header component.
///
/// Handles both:
/// - Unquoted: b/path or a/path
/// - Quoted: "b/path" or "a/path"
///
/// Returns the file path without the a/ or b/ prefix.
pub fn parse_git_path(path: &str) -> String {
    let mut path = path.to_string();

    // Handle quoted paths first.
    if path.starts_with('"') && path.ends_with('"') && path.len() >= 2 {
        path = path[1..path.len() - 1].to_string();
    }

    // Remove leading a/ or b/ prefix.
    if path.starts_with("a/") || path.starts_with("b/") {
        path = path[2..].to_string();
    }

    path
}

/// Extract the 'b/' path from a 'diff --git' line.
///
/// Handles both quoted and unquoted formats:
/// - diff --git a/path b/path
/// - diff --git "a/path with spaces" "b/path with spaces"
///
/// Returns the b/ path component (with or without quotes), or None if parsing fails.
pub fn extract_b_path(line: &str) -> Option<String> {
    // Remove the 'diff --git ' prefix
    if !line.starts_with(GIT_DIFF_HEADER_PREFIX) {
        return None;
    }

    let remainder = line[GIT_DIFF_HEADER_PREFIX.len()..].trim_start();
    if remainder.is_empty() {
        return None;
    }

    // Check if paths are quoted
    if remainder.starts_with('"') {
        // Quoted format: "a/path" "b/path"
        // Find the closing quote of the first quoted string
        let first_quote_end = remainder[1..].find('"')? + 1; // Adjust for slice offset

        // Find the opening quote of the second quoted string
        let second_quote_start = remainder[first_quote_end + 1..].find('"')? + first_quote_end + 1;

        // Find the closing quote of the second quoted string
        let second_quote_end = remainder[second_quote_start + 1..].find('"');
        let second_quote_end = match second_quote_end {
            Some(end) => end + second_quote_start + 1,
            None => remainder.len(), // No closing quote - take rest of line
        };

        // Extract the b/ path (second quoted string)
        Some(remainder[second_quote_start + 1..second_quote_end].to_string())
    } else {
        // Unquoted format: a/path b/path
        let parts: Vec<&str> = remainder.split_whitespace().collect();
        if parts.len() >= 2 {
            Some(parts[1].to_string()) // Last part is the b/ path
        } else {
            None
        }
    }
}

/// Represents a file change in a diff.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffFile {
    /// File path
    pub path: String,
    /// Status: "added", "modified", "deleted"
    pub status: String,
    /// List of hunk texts
    pub hunks: Vec<String>,
}

/// Parser for git diff output.
#[derive(Debug, Clone)]
pub struct DiffParser {
    /// If true, track parse error count
    warn_on_parse_error: bool,
}

impl DiffParser {
    /// Create a new DiffParser.
    pub fn new(warn_on_parse_error: bool) -> Self {
        Self {
            warn_on_parse_error,
        }
    }

    /// Parse git diff output into file changes.
    pub fn parse(&self, diff_output: &str) -> Vec<DiffFile> {
        let mut files: Vec<DiffFile> = Vec::new();

        let mut current_file: Option<String> = None;
        let mut current_status: Option<String> = None;
        let mut current_hunks: Vec<String> = Vec::new();
        let mut current_hunk_lines: Vec<String> = Vec::new();
        let mut _parse_error_count: usize = 0;

        for line in diff_output.split('\n') {
            // New file start
            if line.starts_with("diff --git") {
                // Yield previous file if exists
                if let Some(ref file_path) = current_file {
                    if !current_hunk_lines.is_empty() {
                        current_hunks.push(current_hunk_lines.join("\n"));
                    }
                    files.push(DiffFile {
                        path: file_path.clone(),
                        status: current_status.clone().unwrap_or_else(|| "modified".to_string()),
                        hunks: current_hunks.clone(),
                    });
                }

                // Parse the file path from "diff --git a/path b/path" or quoted variant
                let b_path = extract_b_path(line);
                if let Some(ref path) = b_path {
                    current_file = Some(parse_git_path(path));
                } else {
                    if self.warn_on_parse_error {
                        eprintln!(
                            "Warning: Failed to parse diff header: {}...",
                            &line[..line.len().min(60)]
                        );
                    }
                    _parse_error_count += 1;
                    current_file = None; // Explicitly set to None to prevent content association
                }
                current_status = None;
                current_hunks = Vec::new();
                current_hunk_lines = Vec::new();
            }
            // File status
            else if line.starts_with("new file") {
                current_status = Some("added".to_string());
            } else if line.starts_with("deleted file") {
                current_status = Some("deleted".to_string());
            } else if line.starts_with("old mode") || line.starts_with("new mode") {
                // Ignore mode changes
            }
            // Hunk header
            else if line.starts_with("@@") {
                if !current_hunk_lines.is_empty() {
                    current_hunks.push(current_hunk_lines.join("\n"));
                }
                current_hunk_lines = vec![line.to_string()];
            }
            // Regular diff content
            else if current_file.is_some() {
                current_hunk_lines.push(line.to_string());
            }
        }

        // Yield the last file
        if let Some(ref file_path) = current_file {
            if !current_hunk_lines.is_empty() {
                current_hunks.push(current_hunk_lines.join("\n"));
            }
            files.push(DiffFile {
                path: file_path.clone(),
                status: current_status.unwrap_or_else(|| "modified".to_string()),
                hunks: current_hunks,
            });
        }

        files
    }

    /// Convert a diff to plain text for embedding.
    pub fn parse_diff_to_text(&self, diff_output: &str) -> String {
        let mut parts: Vec<String> = Vec::new();
        for diff_file in self.parse(diff_output) {
            parts.push(format!("File: {} ({})", diff_file.path, diff_file.status));
            for hunk in diff_file.hunks {
                parts.push(hunk);
            }
        }
        parts.join("\n\n")
    }
}

// =============================================================================
// RST (reStructuredText) Parsing
// =============================================================================

/// Valid RST heading underline characters (rank from highest to lowest)
const RST_HEADING_CHARS: &str = "=-~^+*`";

/// Represents a chunk from an RST document.
#[derive(Debug, Clone, PartialEq)]
pub struct RSTChunk {
    /// The text content of the chunk
    pub text: String,
    /// Section path (list of heading texts from root to current)
    pub section_path: Vec<String>,
    /// Directive contents (note, warning, versionadded, etc.)
    pub directives: std::collections::HashMap<String, Vec<String>>,
    /// Temporal tags (deprecated, versionchanged, etc.)
    pub temporal_tags: Vec<String>,
    /// Index of this chunk in the document
    pub chunk_index: u32,
}

/// Find all section boundaries in RST content.
///
/// Returns list of (line_number, heading_text, underline) tuples.
/// Uses only valid RST heading underline characters.
pub fn find_section_boundaries(content: &str) -> Vec<(usize, String, String)> {
    let lines: Vec<&str> = content.lines().collect();
    let mut boundaries = Vec::new();

    for i in 0..lines.len() {
        let line = lines[i];
        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Check if this line could be an underline (previous line was heading)
        if i > 0 {
            let prev_line = lines[i - 1];
            let heading = prev_line.trim();
            let underline = line.trim();

            // Heading and underline must be non-empty
            if heading.is_empty() || underline.is_empty() {
                continue;
            }

            // Underline must be one repeated valid heading character
            let underline_chars: Vec<char> = underline.chars().collect();
            if underline_chars.is_empty() || !underline_chars.iter().all(|&c| c == underline_chars[0]) {
                continue;
            }

            if !RST_HEADING_CHARS.contains(underline_chars[0]) {
                continue;
            }

            // Underline must be at least as long as the heading text
            if underline.len() >= heading.len() {
                boundaries.push((i, heading.to_string(), underline.to_string()));
            }
        }
    }

    boundaries
}

/// Get heading level from underline character.
pub fn get_heading_level(underline: &str) -> u32 {
    if underline.is_empty() {
        return 0;
    }
    let char = underline.chars().next().unwrap();
    match char {
        '=' => 1,
        '-' => 2,
        '~' => 3,
        '^' => 4,
        '+' => 5,
        '*' => 6,
        '`' => 7,
        _ => 0,
    }
}

/// Split content into chunks based on section boundaries.
///
/// Returns list of (start_line, end_line, section_path) tuples.
pub fn chunk_by_sections(content: &str) -> Vec<(usize, usize, Vec<String>)> {
    let lines: Vec<&str> = content.lines().collect();
    let boundaries = find_section_boundaries(content);

    if boundaries.is_empty() {
        // No sections found, return whole content as single chunk
        return vec![(0, lines.len(), vec![])];
    }

    let mut chunks = Vec::new();
    let mut current_path: Vec<(String, u32)> = Vec::new(); // (heading, level)

    // Preserve preamble content before the first section heading.
    let first_heading_line = boundaries[0].0 - 1;
    if first_heading_line > 0 {
        chunks.push((0, first_heading_line, vec![]));
    }

    for i in 0..boundaries.len() {
        let (line_num, heading, underline) = &boundaries[i];
        let level = get_heading_level(underline);

        // Remove any sections at same or higher level from path
        while let Some((_, prev_level)) = current_path.last() {
            if *prev_level >= level {
                current_path.pop();
            } else {
                break;
            }
        }

        // Add current section
        current_path.push((heading.clone(), level));

        // Determine chunk boundaries
        let start_line = *line_num - 1;
        let end_line = if i + 1 < boundaries.len() {
            boundaries[i + 1].0 - 1
        } else {
            lines.len()
        };

        // Extract section path as list of headings
        let section_path: Vec<String> = current_path.iter().map(|(h, _)| h.clone()).collect();

        chunks.push((start_line, end_line, section_path));
    }

    chunks
}

/// Get text content for a chunk from line range.
fn get_chunk_text(lines: &[&str], start: usize, end: usize) -> String {
    lines[start..end].join("\n").trim().to_string()
}

/// Extract directive content from chunk text.
pub fn extract_directives(chunk_text: &str) -> (std::collections::HashMap<String, Vec<String>>, Vec<String>) {
    let mut directives: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    let mut temporal_tags: Vec<String> = Vec::new();

    // Note directive pattern
    let note_re = regex::Regex::new(r"(?m)^\.\. note::(?:\s+(.*))?$\n((?:[ \t]+.+\n?)+)").unwrap();
    for cap in note_re.captures_iter(chunk_text) {
        if let Some(content) = cap.get(2) {
            directives
                .entry("note".to_string())
                .or_insert_with(Vec::new)
                .push(content.as_str().trim().to_string());
        }
    }

    // Warning directive pattern
    let warning_re = regex::Regex::new(r"(?m)^\.\. warning::(?:\s+(.*))?$\n((?:[ \t]+.+\n?)+)").unwrap();
    for cap in warning_re.captures_iter(chunk_text) {
        if let Some(content) = cap.get(2) {
            directives
                .entry("warning".to_string())
                .or_insert_with(Vec::new)
                .push(content.as_str().trim().to_string());
        }
    }

    // Versionadded directive pattern (with optional version arg and content directly after)
    let versionadded_re = regex::Regex::new(r"(?m)^\.\. versionadded::(?:\s+(.*))?$\n\n?((?:[ \t]+.+\n?)+)").unwrap();
    for cap in versionadded_re.captures_iter(chunk_text) {
        if let Some(content) = cap.get(2) {
            directives
                .entry("versionadded".to_string())
                .or_insert_with(Vec::new)
                .push(content.as_str().trim().to_string());
        }
    }

    // Check for deprecated directive (temporal tag)
    let deprecated_re = regex::Regex::new(r"(?m)^\.\. deprecated::\s*(.*?)(?:\n|$)").unwrap();
    if deprecated_re.is_match(chunk_text) {
        temporal_tags.push("deprecated".to_string());
    }

    // Check for versionchanged directive (temporal tag)
    let versionchanged_re = regex::Regex::new(r"(?m)^\.\. versionchanged::\s*(.*?)(?:\n|$)").unwrap();
    if versionchanged_re.is_match(chunk_text) {
        temporal_tags.push("versionchanged".to_string());
    }

    (directives, temporal_tags)
}

/// Check for temporal directives and return tags.
pub fn check_temporal_tags(text: &str) -> Vec<String> {
    let mut tags = Vec::new();

    // Deprecated
    let deprecated_re = regex::Regex::new(r"(?m)^\.\. deprecated::\s*(.*?)(?:\n|$)").unwrap();
    if deprecated_re.is_match(text) {
        tags.push("deprecated".to_string());
    }

    // Version changed
    let versionchanged_re = regex::Regex::new(r"(?m)^\.\. versionchanged::\s*(.*?)(?:\n|$)").unwrap();
    if versionchanged_re.is_match(text) {
        tags.push("versionchanged".to_string());
    }

    tags
}

/// Parse RST content and return list of chunks with metadata.
///
/// Handles malformed/empty content gracefully by returning empty list.
pub fn parse_rst_content(content: &str) -> Vec<RSTChunk> {
    if content.trim().is_empty() {
        return vec![];
    }

    let mut chunks: Vec<RSTChunk> = Vec::new();

    // Handle common docutils parsing issues - normalize line endings
    let normalized = content.replace("\r\n", "\n");

    let lines: Vec<&str> = normalized.lines().collect();
    let section_chunks = chunk_by_sections(&normalized);

    for (idx, (start, end, section_path)) in section_chunks.iter().enumerate() {
        let chunk_text = get_chunk_text(&lines, *start, *end);

        if chunk_text.trim().is_empty() {
            continue;
        }

        // Extract directives
        let (directives, temporal_tags) = extract_directives(&chunk_text);

        let chunk = RSTChunk {
            text: chunk_text,
            section_path: section_path.clone(),
            directives,
            temporal_tags,
            chunk_index: idx as u32,
        };
        chunks.push(chunk);
    }

    chunks
}

/// Extract metadata from a chunk for indexing.
pub fn get_chunk_metadata(chunk: &RSTChunk) -> std::collections::HashMap<String, serde_json::Value> {
    use serde_json::{json, Value};

    let mut metadata = std::collections::HashMap::new();

    metadata.insert("chunk_index".to_string(), json!(chunk.chunk_index));
    metadata.insert("section_path".to_string(), json!(chunk.section_path));
    metadata.insert("has_directives".to_string(), json!(!chunk.directives.is_empty()));
    metadata.insert("has_temporal_tags".to_string(), json!(!chunk.temporal_tags.is_empty()));

    // Add directive content if present
    if !chunk.directives.is_empty() {
        // Convert directives to JSON value
        let directives_json: std::collections::HashMap<String, Vec<Value>> = chunk
            .directives
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().map(|s| json!(s)).collect()))
            .collect();
        metadata.insert("directives".to_string(), json!(directives_json));
    }

    // Add temporal tags if present
    if !chunk.temporal_tags.is_empty() {
        metadata.insert("temporal_tags".to_string(), json!(chunk.temporal_tags));
    }

    // Add section depth
    if !chunk.section_path.is_empty() {
        metadata.insert("section_depth".to_string(), json!(chunk.section_path.len()));
    }

    metadata
}

// =============================================================================
// Minimal Compile-Only Functions (Scaffolding)
// =============================================================================

/// Returns the version of the te-core library.
/// This is a minimal compile-only function to verify the build works.
pub fn get_version() -> &'static str {
    "0.1.0-scaffold"
}

/// Health check function that returns a greeting.
/// Used to verify PyO3 and clap wiring are functional.
pub fn health_check() -> String {
    "te-core: OK".to_string()
}

/// Placeholder function demonstrating the module structure.
/// This will be expanded in subsequent tickets.
pub fn placeholder() -> u32 {
    42
}

// =============================================================================
// Minimal Backend Abstractions for Embedding and Vector Storage
// =============================================================================

use std::sync::Arc;

/// Error type for backend operations.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    /// Embedding generation failed
    EmbeddingError(String),
    /// Vector storage operation failed
    StorageError(String),
    /// Search operation failed
    SearchError(String),
    /// Configuration error
    ConfigError(String),
    /// Document not found
    NotFound(String),
    /// Invalid input
    InvalidInput(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::EmbeddingError(msg) => write!(f, "Embedding error: {}", msg),
            BackendError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            BackendError::SearchError(msg) => write!(f, "Search error: {}", msg),
            BackendError::ConfigError(msg) => write!(f, "Config error: {}", msg),
            BackendError::NotFound(msg) => write!(f, "Not found: {}", msg),
            BackendError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

/// A vector embedding with optional metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    /// The vector values
    pub values: Vec<f32>,
    /// Optional metadata associated with this embedding
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Embedding {
    /// Create a new embedding with values and optional metadata.
    pub fn new(values: Vec<f32>, metadata: Option<HashMap<String, serde_json::Value>>) -> Self {
        Self {
            values,
            metadata: metadata.unwrap_or_default(),
        }
    }

    /// Get the dimensionality of this embedding.
    pub fn dimensions(&self) -> usize {
        self.values.len()
    }

    /// Compute cosine similarity with another embedding.
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.values.is_empty() || other.values.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

/// A document with content, ID, and metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    /// Unique document identifier
    pub id: String,
    /// Document content (text)
    pub content: String,
    /// Document metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Document {
    /// Create a new document.
    pub fn new(id: String, content: String, metadata: Option<HashMap<String, serde_json::Value>>) -> Self {
        Self {
            id,
            content,
            metadata: metadata.unwrap_or_default(),
        }
    }
}

/// A search result with document and score.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// The matched document
    pub document: Document,
    /// Similarity score (higher is better)
    pub score: f32,
}

/// Configuration for backend initialization.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendConfig {
    /// Embedding dimensions
    pub embedding_dimensions: usize,
    /// Optional backend-specific configuration
    pub backend_type: BackendType,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            embedding_dimensions: 384,
            backend_type: BackendType::InMemory,
        }
    }
}

/// Type of backend to use.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// In-memory backend for testing/local development
    InMemory,
    /// Python interop (placeholder for future)
    Python,
    /// Remote backend (placeholder for future)
    Remote(String),
}

/// Trait for text embedding generation.
pub trait Embedder: Send + Sync {
    /// Generate embeddings for the given texts.
    fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, BackendError>;

    /// Get the dimensionality of embeddings produced by this embedder.
    fn dimensions(&self) -> usize;
}

/// Trait for vector storage and retrieval.
pub trait VectorStore: Send + Sync {
    /// Add documents to the store.
    fn add(&self, documents: Vec<Document>, embeddings: Vec<Embedding>) -> Result<(), BackendError>;

    /// Search for similar documents.
    fn search(&self, query_embedding: &Embedding, top_k: usize) -> Result<Vec<SearchResult>, BackendError>;

    /// Delete a document by ID.
    fn delete(&self, doc_id: &str) -> Result<(), BackendError>;

    /// Get a document by ID.
    fn get(&self, doc_id: &str) -> Result<Document, BackendError>;

    /// Get the number of documents in the store.
    fn count(&self) -> usize;

    /// Filter documents by metadata.
    fn filter(&self, filter: &HashMap<String, serde_json::Value>) -> Result<Vec<Document>, BackendError>;
}

/// Combined embedder and vector store interface.
pub trait EmbeddingBackend: Send + Sync {
    /// Generate embeddings and add documents to the store in one step.
    fn embed_and_store(&self, documents: Vec<Document>) -> Result<(), BackendError>;

    /// Search the store with a text query.
    fn search_text(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, BackendError>;

    /// Get the embedder component.
    fn embedder(&self) -> Arc<dyn Embedder>;

    /// Get the vector store component.
    fn vector_store(&self) -> Arc<dyn VectorStore>;
}

// =============================================================================
// In-Memory Implementations
// =============================================================================

use std::sync::RwLock;

/// In-memory embedder implementation (random embeddings for testing).
pub struct InMemoryEmbedder {
    dimensions: usize,
}

impl InMemoryEmbedder {
    /// Create a new in-memory embedder with specified dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl Embedder for InMemoryEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Embedding>, BackendError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Use a deterministic seed based on text content for reproducibility
        let mut embeddings = Vec::with_capacity(texts.len());
        for (idx, text) in texts.iter().enumerate() {
            let mut values = Vec::with_capacity(self.dimensions);
            // Generate pseudo-random but deterministic values based on text
            let seed = text.len() * 1000 + idx;
            for i in 0..self.dimensions {
                let val = ((seed + i * 31) as f32).sin() / (i as f32 + 1.0).abs();
                values.push(val);
            }

            // Normalize to unit vector
            let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                values = values.iter().map(|x| x / norm).collect();
            }

            embeddings.push(Embedding::new(values, None));
        }

        Ok(embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// In-memory vector store implementation.
pub struct InMemoryVectorStore {
    documents: RwLock<HashMap<String, Document>>,
    embeddings: RwLock<HashMap<String, Embedding>>,
}

impl InMemoryVectorStore {
    /// Create a new in-memory vector store.
    pub fn new() -> Self {
        Self {
            documents: RwLock::new(HashMap::new()),
            embeddings: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStore for InMemoryVectorStore {
    fn add(&self, documents: Vec<Document>, embeddings: Vec<Embedding>) -> Result<(), BackendError> {
        if documents.len() != embeddings.len() {
            return Err(BackendError::InvalidInput(
                "Number of documents must match number of embeddings".to_string(),
            ));
        }

        let mut docs = self.documents.write().map_err(|_| {
            BackendError::StorageError("Failed to acquire write lock".to_string())
        })?;
        let mut embeds = self.embeddings.write().map_err(|_| {
            BackendError::StorageError("Failed to acquire write lock".to_string())
        })?;

        for (doc, emb) in documents.into_iter().zip(embeddings.into_iter()) {
            let doc_id = doc.id.clone();
            docs.insert(doc_id.clone(), doc);
            embeds.insert(doc_id, emb);
        }

        Ok(())
    }

    fn search(&self, query_embedding: &Embedding, top_k: usize) -> Result<Vec<SearchResult>, BackendError> {
        let docs = self.documents.read().map_err(|_| {
            BackendError::SearchError("Failed to acquire read lock".to_string())
        })?;
        let embeds = self.embeddings.read().map_err(|_| {
            BackendError::SearchError("Failed to acquire read lock".to_string())
        })?;

        let mut results: Vec<SearchResult> = docs
            .iter()
            .filter_map(|(id, doc)| {
                embeds.get(id).map(|emb| {
                    let score = query_embedding.cosine_similarity(emb);
                    SearchResult {
                        document: doc.clone(),
                        score,
                    }
                })
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k
        results.truncate(top_k);
        Ok(results)
    }

    fn delete(&self, doc_id: &str) -> Result<(), BackendError> {
        let mut docs = self.documents.write().map_err(|_| {
            BackendError::StorageError("Failed to acquire write lock".to_string())
        })?;
        let mut embeds = self.embeddings.write().map_err(|_| {
            BackendError::StorageError("Failed to acquire write lock".to_string())
        })?;

        docs.remove(doc_id);
        embeds.remove(doc_id);

        Ok(())
    }

    fn get(&self, doc_id: &str) -> Result<Document, BackendError> {
        let docs = self.documents.read().map_err(|_| {
            BackendError::StorageError("Failed to acquire read lock".to_string())
        })?;

        docs.get(doc_id)
            .cloned()
            .ok_or_else(|| BackendError::NotFound(format!("Document {} not found", doc_id)))
    }

    fn count(&self) -> usize {
        self.documents.read()
            .map(|docs| docs.len())
            .unwrap_or(0)
    }

    fn filter(&self, filter: &HashMap<String, serde_json::Value>) -> Result<Vec<Document>, BackendError> {
        let docs = self.documents.read().map_err(|_| {
            BackendError::StorageError("Failed to acquire read lock".to_string())
        })?;

        let results: Vec<Document> = docs
            .values()
            .filter(|doc| {
                filter.iter().all(|(key, value)| {
                    doc.metadata.get(key) == Some(value)
                })
            })
            .cloned()
            .collect();

        Ok(results)
    }
}

/// Combined in-memory backend for embedder + vector store.
pub struct InMemoryBackend {
    embedder: Arc<InMemoryEmbedder>,
    store: Arc<InMemoryVectorStore>,
}

impl InMemoryBackend {
    /// Create a new in-memory backend.
    pub fn new(dimensions: usize) -> Self {
        Self {
            embedder: Arc::new(InMemoryEmbedder::new(dimensions)),
            store: Arc::new(InMemoryVectorStore::new()),
        }
    }

    /// Create from config.
    pub fn from_config(config: &BackendConfig) -> Self {
        Self::new(config.embedding_dimensions)
    }
}

impl EmbeddingBackend for InMemoryBackend {
    fn embed_and_store(&self, documents: Vec<Document>) -> Result<(), BackendError> {
        let texts: Vec<String> = documents.iter().map(|d| d.content.clone()).collect();
        let embeddings = self.embedder.embed(&texts)?;
        self.store.add(documents, embeddings)
    }

    fn search_text(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, BackendError> {
        let embeddings = self.embedder.embed(&[query.to_string()])?;
        let query_embedding = &embeddings[0];
        self.store.search(query_embedding, top_k)
    }

    fn embedder(&self) -> Arc<dyn Embedder> {
        self.embedder.clone()
    }

    fn vector_store(&self) -> Arc<dyn VectorStore> {
        self.store.clone()
    }
}

/// Build indexable documents from files discovered under a root path.
///
/// This shares file scanning/chunking/doc-id logic with the Rust CLI so the
/// binary stays as a thin command adapter.
pub fn build_documents_from_files(
    root_path: &Path,
    extensions: Option<&HashSet<String>>,
    exclude_patterns: Option<&HashSet<String>>,
) -> Result<Vec<Document>, std::io::Error> {
    use serde_json::json;

    let mut documents: Vec<Document> = Vec::new();

    for file_path in scan_files(root_path, extensions, exclude_patterns) {
        let content = std::fs::read_to_string(&file_path)?;
        let file_type = file_path
            .extension()
            .map(|ext| format!(".{}", ext.to_string_lossy()))
            .unwrap_or_else(String::new);

        let mut base_metadata = HashMap::new();
        base_metadata.insert(
            "source".to_string(),
            json!(file_path.to_string_lossy().to_string()),
        );
        base_metadata.insert("type".to_string(), json!(file_type.clone()));

        let chunks: Vec<(String, HashMap<String, serde_json::Value>)> = if file_type == ".rst" {
            let rst_chunks = parse_rst_content(&content);
            if rst_chunks.is_empty() {
                vec![(content.clone(), HashMap::from([("chunk_index".to_string(), json!(0))]))]
            } else {
                rst_chunks
                    .into_iter()
                    .map(|chunk| {
                        let metadata = get_chunk_metadata(&chunk);
                        (chunk.text, metadata)
                    })
                    .collect()
            }
        } else {
            vec![(content.clone(), HashMap::from([("chunk_index".to_string(), json!(0))]))]
        };

        let path_str = file_path.to_string_lossy().to_string();
        for (fallback_chunk_index, (chunk_text, chunk_metadata)) in chunks.into_iter().enumerate() {
            let (metadata, chunk_index) = normalize_chunk_metadata(
                &base_metadata,
                &chunk_metadata,
                fallback_chunk_index as u32,
            );
            let doc_id = build_file_doc_id(&path_str, chunk_index);
            documents.push(Document::new(doc_id, chunk_text, Some(metadata)));
        }
    }

    Ok(documents)
}

// =============================================================================
// Deterministic Doc-ID Generation
// =============================================================================

/// Build deterministic doc ID for file content chunks.
///
/// For chunk_index == 0, uses the path directly.
/// For chunk_index > 0, appends "#chunk:{chunk_index}" to the path.
///
/// Returns the first 16 hex characters of SHA256 hash.
pub fn build_file_doc_id(path: &str, chunk_index: u32) -> String {
    use sha2::{Sha256, Digest};

    let doc_id_input = if chunk_index == 0 {
        path.to_string()
    } else {
        format!("{}#chunk:{}", path, chunk_index)
    };

    let mut hasher = Sha256::new();
    hasher.update(doc_id_input.as_bytes());
    let result = hasher.finalize();

    // Return first 16 hex characters
    format!("{:x}", result)[..16].to_string()
}

/// Return canonical and legacy ID input strings for a path.
///
/// This is used for deletion safety to ensure we can find docs
/// regardless of whether they were indexed with relative or absolute paths.
pub fn get_doc_id_inputs(path: &str, repo_root: &Path) -> Vec<String> {
    let mut doc_id_inputs = vec![path.to_string()];

    // If path is not absolute, also try the resolved absolute path
    let path_obj = Path::new(path);
    if !path_obj.is_absolute() {
        let absolute_path = normalize_absolute_path(&repo_root.join(path_obj));
        let resolved = absolute_path.to_string_lossy().to_string();
        if resolved != path {
            doc_id_inputs.push(resolved);
        }
    }

    doc_id_inputs
}

/// Normalize chunk metadata by merging base and chunk metadata,
/// and validate/normalize the chunk_index field.
///
/// Returns a tuple of (normalized_metadata, chunk_index).
pub fn normalize_chunk_metadata(
    base_metadata: &HashMap<String, serde_json::Value>,
    chunk_metadata: &HashMap<String, serde_json::Value>,
    fallback_chunk_index: u32,
) -> (HashMap<String, serde_json::Value>, u32) {
    use serde_json::Value;

    let mut metadata = base_metadata.clone();
    metadata.extend(chunk_metadata.clone());

    let chunk_index_value = metadata.get("chunk_index");

    let chunk_index = match chunk_index_value {
        Some(Value::Number(n)) => {
            if let Some(idx) = n.as_u64() {
                // idx is always >= 0 since it's u64
                idx as u32
            } else {
                fallback_chunk_index
            }
        }
        // Bool is treated as invalid (matches Python isinstance(..., bool) check)
        Some(Value::Bool(_)) | None => fallback_chunk_index,
        _ => fallback_chunk_index,
    };

    // Always ensure chunk_index is set in metadata
    metadata.insert("chunk_index".to_string(), Value::Number(chunk_index.into()));

    (metadata, chunk_index)
}

// =============================================================================
// PyO3 Python Bindings
// =============================================================================

#[cfg(feature = "python")]
mod pyo3_bindings {
    // PyO3 bindings are defined in rust/src/lib.rs instead
    // This module re-exports te-core functions for the Python module
}

// =============================================================================
// Clap CLI Bindings
// =============================================================================

use clap::Parser;

/// Minimal CLI for testing the clap binary entrypoint.
#[derive(Parser, Debug)]
#[command(name = "te-core")]
#[command(version = "0.1.0-scaffold")]
#[command(about = "Town Elder Core CLI (scaffolding)", long_about = None)]
pub struct Cli {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Subcommand to run
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Parser, Debug)]
enum Commands {
    /// Run the health check
    Health,
    /// Print version info
    Version,
    /// Run the placeholder command
    Placeholder,
}

impl Cli {
    /// Execute the CLI logic.
    pub fn run(self) {
        if self.verbose {
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
                .init();
        }

        match self.command {
            Some(Commands::Health) => {
                println!("{}", health_check());
            }
            Some(Commands::Version) => {
                println!("{}", get_version());
            }
            Some(Commands::Placeholder) => {
                println!("{}", placeholder());
            }
            None => {
                // No subcommand: print help
                println!("te-core CLI (scaffolding)");
                println!("Use --help for more information");
            }
        }
    }
}

/// Entry point for the clap binary.
/// Can be run via: `cargo run --manifest-path rust/Cargo.toml --bin te-core`
pub fn main() {
    let cli = Cli::parse();
    cli.run();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(prefix: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "te-core-{prefix}-{}-{timestamp}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("create temp directory");
        directory
    }

    #[test]
    fn test_version() {
        assert_eq!(get_version(), "0.1.0-scaffold");
    }

    #[test]
    fn test_health_check() {
        assert_eq!(health_check(), "te-core: OK");
    }

    #[test]
    fn test_placeholder() {
        assert_eq!(placeholder(), 42);
    }

    #[test]
    fn test_scan_files_matches_default_extensions_and_sorted_order() {
        let root = make_temp_dir("scan-defaults");
        fs::create_dir_all(root.join("src")).expect("create src");
        fs::write(root.join("src/a.py"), "print('a')").expect("write a.py");
        fs::write(root.join("src/c.rst"), "Title\n=====\n").expect("write c.rst");
        fs::write(root.join("src/b.md"), "# note").expect("write b.md");
        fs::write(root.join("src/ignore.txt"), "ignore").expect("write txt");

        let files = scan_files(&root, None, None);
        let names: Vec<String> = files
            .iter()
            .map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();

        assert_eq!(names, vec!["a.py", "b.md", "c.rst"]);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn test_scan_files_applies_default_and_custom_excludes() {
        let root = make_temp_dir("scan-excludes");
        fs::create_dir_all(root.join("_build")).expect("create _build");
        fs::create_dir_all(root.join("custom")).expect("create custom");
        fs::write(root.join("_build/generated.py"), "generated").expect("write generated.py");
        fs::write(root.join("custom/skip.py"), "skip").expect("write skip.py");
        fs::write(root.join("keep.py"), "keep").expect("write keep.py");

        let custom_excludes = HashSet::from(["custom".to_string()]);
        let files = scan_files(&root, None, Some(&custom_excludes));
        let names: Vec<String> = files
            .iter()
            .map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();

        assert_eq!(names, vec!["keep.py"]);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn test_get_doc_id_inputs_normalizes_relative_paths() {
        let repo_root = Path::new("/Users/testuser/repo");
        let inputs = get_doc_id_inputs("src/../src/main.py", repo_root);
        assert!(inputs.contains(&"src/../src/main.py".to_string()));
        assert!(inputs.contains(&"/Users/testuser/repo/src/main.py".to_string()));
    }

    #[test]
    fn test_build_documents_from_files_creates_rst_chunks() {
        let root = make_temp_dir("build-documents");
        fs::create_dir_all(root.join("docs")).expect("create docs");
        fs::write(
            root.join("docs/guide.rst"),
            "Guide\n=====\n\nIntro.\n\nSection\n-------\n\nBody.\n",
        )
        .expect("write rst");

        let docs = build_documents_from_files(&root, None, None).expect("build docs");
        assert!(docs.len() >= 2);
        assert!(docs.iter().all(|doc| doc.metadata.contains_key("chunk_index")));
        assert!(docs
            .iter()
            .any(|doc| doc.metadata.get("type") == Some(&serde_json::json!(".rst"))));
        fs::remove_dir_all(root).ok();
    }

    // =============================================================================
    // RST Parsing Tests
    // =============================================================================

    #[test]
    fn test_heading_level_equal_sign() {
        assert_eq!(get_heading_level("="), 1);
        assert_eq!(get_heading_level("==="), 1);
    }

    #[test]
    fn test_heading_level_hyphen() {
        assert_eq!(get_heading_level("-"), 2);
        assert_eq!(get_heading_level("---"), 2);
    }

    #[test]
    fn test_heading_level_tilde() {
        assert_eq!(get_heading_level("~"), 3);
        assert_eq!(get_heading_level("~~~"), 3);
    }

    #[test]
    fn test_heading_level_caret() {
        assert_eq!(get_heading_level("^"), 4);
        assert_eq!(get_heading_level("^^^"), 4);
    }

    #[test]
    fn test_heading_level_plus() {
        assert_eq!(get_heading_level("+"), 5);
        assert_eq!(get_heading_level("+++"), 5);
    }

    #[test]
    fn test_find_section_boundaries_single() {
        let content = "Title\n=====\n\nSome content here.";
        let boundaries = find_section_boundaries(content);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].1, "Title");
    }

    #[test]
    fn test_find_section_boundaries_multiple() {
        let content = "Title\n=====\n\nSection One\n-----------\n\nContent one.\n\nSection Two\n-----------\n\nContent two.";
        let boundaries = find_section_boundaries(content);
        // Should find Title + 2 sections = 3 boundaries
        assert_eq!(boundaries.len(), 3);
    }

    #[test]
    fn test_find_section_boundaries_no_sections() {
        let content = "Just some plain text.\nNo headings here.";
        let boundaries = find_section_boundaries(content);
        assert!(boundaries.is_empty());
    }

    #[test]
    fn test_find_section_boundaries_mixed_underline() {
        let content = "Title\n=-=\n\nBody text.";
        let boundaries = find_section_boundaries(content);
        assert!(boundaries.is_empty());
    }

    #[test]
    fn test_chunk_single_section() {
        let content = "Title\n=====\n\nThis is the content.";
        let chunks = chunk_by_sections(content);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_multiple_sections() {
        let content = "Main Title\n============\n\nSection A\n---------\n\nContent A.\n\nSection B\n---------\n\nContent B.";
        let chunks = chunk_by_sections(content);
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_extract_note_directive() {
        let text = "Section\n=======\n\nSome text.\n\n.. note::\n\n   This is a note.";
        let (directives, _) = extract_directives(text);
        assert!(directives.contains_key("note"));
    }

    #[test]
    fn test_extract_warning_directive() {
        let text = "Section\n=======\n\n.. warning::\n\n   This is a warning.";
        let (directives, _) = extract_directives(text);
        assert!(directives.contains_key("warning"));
    }

    #[test]
    fn test_extract_versionadded_directive() {
        let text = "Section\n=======\n\n.. versionadded:: 2.0\n\n   Added in version 2.0.";
        let (directives, _) = extract_directives(text);
        assert!(directives.contains_key("versionadded"));
    }

    #[test]
    fn test_check_temporal_tags_deprecated() {
        let text = "Section\n=======\n\n.. deprecated:: 1.5\n\n   Use new_function instead.";
        let tags = check_temporal_tags(text);
        assert!(tags.contains(&"deprecated".to_string()));
    }

    #[test]
    fn test_check_temporal_tags_versionchanged() {
        let text = "Section\n=======\n\n.. versionchanged:: 2.0\n\n   Changed in version 2.0.";
        let tags = check_temporal_tags(text);
        assert!(tags.contains(&"versionchanged".to_string()));
    }

    #[test]
    fn test_check_temporal_tags_none() {
        let text = "Section\n=======\n\nJust some content.";
        let tags = check_temporal_tags(text);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_parse_simple_rst() {
        let content = "Title\n=====\n\nThis is content.";
        let chunks = parse_rst_content(content);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_parse_rst_with_section_hierarchy() {
        let content = "Main Title\n============\n\nSection One\n-----------\n\nContent one.";
        let chunks = parse_rst_content(content);
        assert!(!chunks.is_empty());
        assert!(!chunks[0].section_path.is_empty());
    }

    #[test]
    fn test_parse_empty_content() {
        let chunks = parse_rst_content("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_parse_whitespace_only() {
        let chunks = parse_rst_content("   \n\n   ");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_metadata_basic() {
        let chunk = RSTChunk {
            text: "Test content".to_string(),
            section_path: vec!["Section".to_string()],
            directives: std::collections::HashMap::new(),
            temporal_tags: vec![],
            chunk_index: 0,
        };
        let metadata = get_chunk_metadata(&chunk);
        assert_eq!(metadata.get("chunk_index"), Some(&serde_json::json!(0)));
        assert_eq!(metadata.get("section_path"), Some(&serde_json::json!(["Section"])));
        assert_eq!(metadata.get("has_directives"), Some(&serde_json::json!(false)));
        assert_eq!(metadata.get("has_temporal_tags"), Some(&serde_json::json!(false)));
    }

    #[test]
    fn test_chunk_metadata_with_directives() {
        let mut directives = std::collections::HashMap::new();
        directives.insert("note".to_string(), vec!["A note".to_string()]);
        let chunk = RSTChunk {
            text: "Test".to_string(),
            section_path: vec![],
            directives,
            temporal_tags: vec![],
            chunk_index: 0,
        };
        let metadata = get_chunk_metadata(&chunk);
        assert_eq!(metadata.get("has_directives"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn test_chunk_metadata_with_temporal_tags() {
        let chunk = RSTChunk {
            text: "Test".to_string(),
            section_path: vec![],
            directives: std::collections::HashMap::new(),
            temporal_tags: vec!["deprecated".to_string()],
            chunk_index: 0,
        };
        let metadata = get_chunk_metadata(&chunk);
        assert_eq!(metadata.get("has_temporal_tags"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn test_chunk_metadata_section_depth() {
        let chunk = RSTChunk {
            text: "Test".to_string(),
            section_path: vec!["Main".to_string(), "Sub".to_string(), "SubSub".to_string()],
            directives: std::collections::HashMap::new(),
            temporal_tags: vec![],
            chunk_index: 0,
        };
        let metadata = get_chunk_metadata(&chunk);
        assert_eq!(metadata.get("section_depth"), Some(&serde_json::json!(3)));
    }

    // =============================================================================
    // Commit Log Parsing Tests
    // =============================================================================

    #[test]
    fn test_parse_commit_header_basic() {
        let line = "abc123def456\x1fcommit message\x1fTest Author\x1f2024-01-01T00:00:00+00:00\x1e";
        let commit = parse_commit_header(line);
        assert!(commit.is_some());
        let c = commit.unwrap();
        assert_eq!(c.hash, "abc123def456");
        assert_eq!(c.message, "commit message");
        assert_eq!(c.author, "Test Author");
        assert_eq!(c.date, "2024-01-01T00:00:00+00:00");
        assert!(c.files_changed.is_empty());
    }

    #[test]
    fn test_parse_commit_header_with_unicode_subject() {
        // Test with Unicode characters in subject (café naïve)
        let line = "hash123\x1ffeat: café naïve punctuation\x1fAuthor\x1f2024-01-01T00:00:00+00:00\x1e";
        let commit = parse_commit_header(line);
        assert!(commit.is_some());
        let c = commit.unwrap();
        assert_eq!(c.message, "feat: café naïve punctuation");
    }

    #[test]
    fn test_parse_commit_header_with_delimiter_in_subject() {
        // Test with delimiter characters in subject (||, <>, etc.)
        let line = "hash123\x1ffix parser ||| edge case\x1fAuthor\x1f2024-01-01T00:00:00+00:00\x1e";
        let commit = parse_commit_header(line);
        assert!(commit.is_some());
        let c = commit.unwrap();
        assert_eq!(c.message, "fix parser ||| edge case");
    }

    #[test]
    fn test_parse_commit_header_with_special_chars() {
        // Test with various special characters
        let line = "hash123\x1ffeat: test []{}<>|~!@\x1fAuthor\x1f2024-01-01T00:00:00+00:00\x1e";
        let commit = parse_commit_header(line);
        assert!(commit.is_some());
        let c = commit.unwrap();
        assert_eq!(c.message, "feat: test []{}<>|~!@");
    }

    #[test]
    fn test_parse_commit_header_invalid_short_hash() {
        // Test with too short hash
        let line = "abc\x1fmessage\x1fAuthor\x1f2024-01-01T00:00:00+00:00\x1e";
        let commit = parse_commit_header(line);
        assert!(commit.is_none());
    }

    #[test]
    fn test_parse_commit_header_invalid_not_enough_parts() {
        // Test with not enough parts
        let line = "hash123\x1fmessage\x1fAuthor\x1e";
        let commit = parse_commit_header(line);
        assert!(commit.is_none());
    }

    #[test]
    fn test_parse_numstat_line_basic() {
        let line = "10\t5\tsrc/main.rs";
        let file_path = parse_numstat_line(line);
        assert_eq!(file_path, Some("src/main.rs".to_string()));
    }

    #[test]
    fn test_parse_numstat_line_binary_file() {
        // Binary files show "-" for additions/deletions
        let line = "-\t-\tbinary.png";
        let file_path = parse_numstat_line(line);
        assert_eq!(file_path, Some("binary.png".to_string()));
    }

    #[test]
    fn test_parse_numstat_line_empty_path() {
        let line = "10\t5\t";
        let file_path = parse_numstat_line(line);
        assert!(file_path.is_none());
    }

    #[test]
    fn test_parse_numstat_line_dash_path() {
        // Dash is used as a placeholder for binary files
        let line = "10\t5\t-";
        let file_path = parse_numstat_line(line);
        assert!(file_path.is_none());
    }

    #[test]
    fn test_parse_numstat_line_empty_line() {
        let file_path = parse_numstat_line("");
        assert!(file_path.is_none());
    }

    #[test]
    fn test_parse_numstat_line_no_tabs() {
        let file_path = parse_numstat_line("not a numstat line");
        assert!(file_path.is_none());
    }

    #[test]
    fn test_parse_commits_with_numstat_basic() {
        let output = "abc123\x1fFirst commit\x1fAuthor\x1f2024-01-01T00:00:00+00:00\x1e\n\
10\t5\tsrc/main.rs\n\
5\t3\tsrc/lib.rs\n\
def456\x1fSecond commit\x1fAuthor\x1f2024-01-02T00:00:00+00:00\x1e\n\
20\t10\tREADME.md";

        let commits = parse_commits_with_numstat(output);
        assert_eq!(commits.len(), 2);

        let c1 = &commits[0];
        assert_eq!(c1.hash, "abc123");
        assert_eq!(c1.message, "First commit");
        assert_eq!(c1.files_changed.len(), 2);
        assert!(c1.files_changed.contains(&"src/main.rs".to_string()));
        assert!(c1.files_changed.contains(&"src/lib.rs".to_string()));

        let c2 = &commits[1];
        assert_eq!(c2.hash, "def456");
        assert_eq!(c2.message, "Second commit");
        assert_eq!(c2.files_changed.len(), 1);
        assert!(c2.files_changed.contains(&"README.md".to_string()));
    }

    #[test]
    fn test_parse_commits_with_numstat_binary_files() {
        let output = "abc123\x1fAdd binary\x1fAuthor\x1f2024-01-01T00:00:00+00:00\x1e\n\
-\t-\tbinary.png\n\
0\t0\tREADME.md";

        let commits = parse_commits_with_numstat(output);
        assert_eq!(commits.len(), 1);
        // Binary files should still be included
        assert_eq!(commits[0].files_changed.len(), 2);
    }

    #[test]
    fn test_parse_commits_with_numstat_empty_output() {
        let commits = parse_commits_with_numstat("");
        assert!(commits.is_empty());
    }

    #[test]
    fn test_is_commit_header_line() {
        assert!(is_commit_header_line("abc123\x1emessage"));
        assert!(is_commit_header_line("hash\x1fsubject\x1fauthor\x1fdate\x1e"));
        assert!(!is_commit_header_line("10\t5\tfile.rs"));
        assert!(!is_commit_header_line("not a header"));
    }

    // =============================================================================
    // Diff Truncation Tests
    // =============================================================================

    #[test]
    fn test_truncate_diff_under_limit() {
        let diff = "diff content";
        let result = truncate_diff(diff, 100);
        assert_eq!(result, diff);
    }

    #[test]
    fn test_truncate_diff_over_limit() {
        let diff = "a".repeat(200);
        let result = truncate_diff(&diff, 100);
        assert!(result.len() < diff.len());
        assert!(result.contains("[truncated"));
        assert!(result.contains("100"));
    }

    #[test]
    fn test_truncate_diff_exact_limit() {
        // At exactly the limit, should not truncate
        let diff = "a".repeat(100);
        let result = truncate_diff(&diff, 100);
        assert_eq!(result, diff);
    }

    #[test]
    fn test_truncate_diff_zero_limit() {
        let diff = "content";
        let result = truncate_diff(diff, 0);
        assert!(result.contains("[truncated"));
    }

    #[test]
    fn test_is_diff_truncated() {
        assert!(is_diff_truncated("content\n\n[truncated - exceeded 100 byte limit]"));
        assert!(!is_diff_truncated("normal diff content"));
    }

    // =============================================================================
    // Commit Text Assembly Tests
    // =============================================================================

    #[test]
    fn test_assemble_commit_text() {
        let message = "Fix bug";
        let diff_text = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new";
        let result = assemble_commit_text(message, diff_text);
        assert!(result.starts_with("Commit: Fix bug\n\n"));
        assert!(result.contains(diff_text));
    }

    #[test]
    fn test_assemble_commit_text_empty_diff() {
        let message = "Initial commit";
        let diff_text = "";
        let result = assemble_commit_text(message, diff_text);
        assert_eq!(result, "Commit: Initial commit\n\n");
    }

    #[test]
    fn test_append_truncation_note_not_truncated() {
        let text = "Commit: test\n\ndiff content";
        let diff = "normal diff";
        let result = append_truncation_note(text, diff);
        assert_eq!(result, text);
    }

    #[test]
    fn test_append_truncation_note_truncated() {
        let text = "Commit: test\n\ndiff content";
        let diff = "diff content\n\n[truncated - exceeded 100 byte limit]";
        let result = append_truncation_note(text, diff);
        assert!(result.contains("[diff was truncated due to size]"));
    }

}
