//! A deterministic ustar writer (fixed mtime, uid/gid 0, sorted input) and a minimal reader
//! for merging upstream archives.

use anyhow::{Context, Result, bail};

/// Fixed mtime for every member we write: 2026-01-01T00:00:00Z.
pub const MTIME: u64 = 1_767_225_600;

/// One archive member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Member {
    Dir { mode: u32 },
    File { mode: u32, data: Vec<u8> },
    Symlink { target: String },
    HardLink { target: String },
}

/// Builds an archive from `(path, member)` pairs; paths are relative, without a leading `/`.
#[derive(Default)]
pub struct TarWriter {
    out: Vec<u8>,
}

fn octal(field: &mut [u8], v: u64) -> Result<()> {
    let s = format!("{:0width$o}\0", v, width = field.len() - 1);
    if s.len() != field.len() {
        bail!("value {v} does not fit a {}-byte tar field", field.len());
    }
    field.copy_from_slice(s.as_bytes());
    Ok(())
}

impl TarWriter {
    pub fn new() -> Self {
        Self::default()
    }

    fn header(&mut self, path: &str, ty: u8, mode: u32, size: u64, link: &str) -> Result<()> {
        let mut h = [0u8; 512];
        let (prefix, name) = if path.len() <= 100 {
            ("", path)
        } else {
            // ustar: split at a '/' so that name <= 100 and prefix <= 155.
            let split = path[..path.len().min(156)]
                .rfind('/')
                .filter(|&i| path.len() - i - 1 <= 100)
                .with_context(|| format!("path too long for ustar: {path}"))?;
            (&path[..split], &path[split + 1..])
        };
        if link.len() > 100 {
            bail!("link target too long for ustar: {link}");
        }
        h[..name.len()].copy_from_slice(name.as_bytes());
        octal(&mut h[100..108], u64::from(mode & 0o7777))?;
        octal(&mut h[108..116], 0)?;
        octal(&mut h[116..124], 0)?;
        octal(&mut h[124..136], size)?;
        octal(&mut h[136..148], MTIME)?;
        h[156] = ty;
        h[157..157 + link.len()].copy_from_slice(link.as_bytes());
        h[257..263].copy_from_slice(b"ustar\0");
        h[263..265].copy_from_slice(b"00");
        h[265..269].copy_from_slice(b"root");
        h[297..301].copy_from_slice(b"root");
        h[345..345 + prefix.len()].copy_from_slice(prefix.as_bytes());
        h[148..156].fill(b' ');
        let sum: u64 = h.iter().map(|&b| u64::from(b)).sum();
        h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
        self.out.extend_from_slice(&h);
        Ok(())
    }

    /// Appends one member.
    pub fn add(&mut self, path: &str, member: &Member) -> Result<()> {
        match member {
            Member::Dir { mode } => self.header(
                &format!("{}/", path.trim_end_matches('/')),
                b'5',
                *mode,
                0,
                "",
            ),
            Member::File { mode, data } => {
                self.header(path, b'0', *mode, data.len() as u64, "")?;
                self.out.extend_from_slice(data);
                let pad = (512 - data.len() % 512) % 512;
                self.out.extend(std::iter::repeat_n(0u8, pad));
                Ok(())
            }
            Member::Symlink { target } => self.header(path, b'2', 0o777, 0, target),
            Member::HardLink { target } => self.header(path, b'1', 0o644, 0, target),
        }
    }

    /// Finishes the archive (two zero blocks).
    pub fn finish(mut self) -> Vec<u8> {
        self.out.extend_from_slice(&[0u8; 1024]);
        self.out
    }
}

/// Reads every member of an (uncompressed) archive: ustar, GNU long names and pax paths.
pub fn read_all(data: &[u8]) -> Result<Vec<(String, Member)>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut long_name: Option<String> = None;
    let mut long_link: Option<String> = None;
    let cstr = |b: &[u8]| {
        String::from_utf8_lossy(&b[..b.iter().position(|&c| c == 0).unwrap_or(b.len())])
            .into_owned()
    };
    let num = |b: &[u8]| -> Result<u64> {
        let s = cstr(b);
        let s = s.trim();
        if s.is_empty() {
            Ok(0)
        } else {
            u64::from_str_radix(s, 8).with_context(|| format!("bad octal {s:?}"))
        }
    };
    while pos + 512 <= data.len() {
        let h = &data[pos..pos + 512];
        if h.iter().all(|&b| b == 0) {
            break;
        }
        let size = num(&h[124..136])? as usize;
        let ty = h[156];
        let body = data
            .get(pos + 512..pos + 512 + size)
            .context("truncated tar member")?;
        pos += 512 + size.div_ceil(512) * 512;
        match ty {
            b'L' => {
                long_name = Some(cstr(body));
                continue;
            }
            b'K' => {
                long_link = Some(cstr(body));
                continue;
            }
            b'x' => {
                let text = String::from_utf8_lossy(body);
                for rec in text.lines() {
                    if let Some((_, kv)) = rec.split_once(' ')
                        && let Some((k, v)) = kv.split_once('=')
                    {
                        match k {
                            "path" => long_name = Some(v.to_owned()),
                            "linkpath" => long_link = Some(v.to_owned()),
                            _ => {}
                        }
                    }
                }
                continue;
            }
            b'g' => continue,
            _ => {}
        }
        let name = long_name.take().unwrap_or_else(|| {
            let n = cstr(&h[0..100]);
            let p = if &h[257..262] == b"ustar" {
                cstr(&h[345..500])
            } else {
                String::new()
            };
            if p.is_empty() { n } else { format!("{p}/{n}") }
        });
        let link = long_link.take().unwrap_or_else(|| cstr(&h[157..257]));
        let mode = num(&h[100..108])? as u32;
        let path = name
            .trim_start_matches("./")
            .trim_start_matches('/')
            .trim_end_matches('/')
            .to_owned();
        // An image index (ours) is never a member to merge.
        if path.is_empty()
            || path == "."
            || path.as_bytes() == wasmshim_linux::vfs::index::INDEX_NAME
        {
            continue;
        }
        let member = match ty {
            b'0' | 0 | b'7' => Member::File {
                mode,
                data: body.to_vec(),
            },
            b'5' => Member::Dir { mode },
            b'2' => Member::Symlink { target: link },
            b'1' => Member::HardLink {
                target: link
                    .trim_start_matches("./")
                    .trim_start_matches('/')
                    .to_owned(),
            },
            _ => continue, // devices, fifos: not needed in images
        };
        out.push((path, member));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut w = TarWriter::new();
        let long = format!("{}/file", "d".repeat(120));
        w.add("bin", &Member::Dir { mode: 0o755 }).unwrap();
        w.add(
            "bin/x",
            &Member::File {
                mode: 0o755,
                data: b"hello".to_vec(),
            },
        )
        .unwrap();
        w.add("bin/y", &Member::Symlink { target: "x".into() })
            .unwrap();
        w.add(
            &long,
            &Member::File {
                mode: 0o644,
                data: vec![1; 600],
            },
        )
        .unwrap();
        let tar = w.finish();
        let back = read_all(&tar).unwrap();
        assert_eq!(back[0], ("bin".into(), Member::Dir { mode: 0o755 }));
        assert_eq!(
            back[1],
            (
                "bin/x".into(),
                Member::File {
                    mode: 0o755,
                    data: b"hello".to_vec()
                }
            )
        );
        assert_eq!(
            back[2],
            ("bin/y".into(), Member::Symlink { target: "x".into() })
        );
        assert_eq!(back[3].0, long);
    }
}
