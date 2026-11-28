//! The host side of an accelerator's token link.
//!
//! The accelerator takes token ids and returns token ids: its `prompt` port
//! carries at most [`PROMPT_MAX`] ids per event, its `token` port emits one id
//! per generated token, and its control core runs the sampler (greedy). The
//! host owns everything textual: the tokenizer, the chat template and the
//! tool-call parse. The physical link is out of scope; this is its framing as
//! lines over any byte stream (TCP to a software model, a UART or a PCIe
//! mailbox):
//!
//! ```text
//! host -> chip   R                         reset: a new sequence (the KV cache is cleared)
//! host -> chip   P <n> <id> ... <id>       up to 64 prompt ids, appended in order
//! host -> chip   G <max_new> <stop>[,<stop>...]
//! chip -> host   T <id>                    one per generated token (a stop id is sent, then D)
//! chip -> host   D <eos|max> <n>           done, n tokens generated
//! chip -> host   E <reason>                refused (the sequence is gone; send R)
//! ```

use std::io::{self, BufRead, BufReader, Write};

/// The most ids one `prompt` event carries (the chip's `prompt` port).
pub const PROMPT_MAX: usize = 64;

/// Why a generation on the accelerator link failed.
#[derive(Debug)]
pub enum LinkError {
    /// The byte stream failed or closed, or no token arrived within the
    /// stream's read timeout.
    Io(io::Error),
    /// The chip answered E.
    Refused(String),
    /// A line that is not the link protocol, more tokens than asked for, or a
    /// count that does not match the tokens received.
    Protocol(String),
}

impl std::fmt::Display for LinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkError::Io(e) => write!(f, "link: {e}"),
            LinkError::Refused(r) => write!(f, "the accelerator refused: {r}"),
            LinkError::Protocol(p) => write!(f, "link protocol: {p}"),
        }
    }
}

impl std::error::Error for LinkError {}

impl From<io::Error> for LinkError {
    fn from(e: io::Error) -> Self {
        LinkError::Io(e)
    }
}

/// What one generation produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    /// The generated token ids in order, including the stop id when one ended the run.
    pub tokens: Vec<u32>,
    /// True when a stop id ended it, false when `max_new` did.
    pub stopped: bool,
}

/// Sends `prompt` (chunked to the chip's port) and reads the generated tokens.
/// `on_token` sees each id as it arrives (streaming to a person is possible).
///
/// Blocks until the accelerator reports that it is done. A read timeout on
/// the stream bounds the wait for each token.
///
/// # Errors
/// [`LinkError::Protocol`] for an empty prompt or no stop id (nothing is
/// sent), and for an answer that breaks the protocol; [`LinkError::Refused`]
/// when the accelerator answers `E`; [`LinkError::Io`] when the stream fails.
///
/// ```
/// use std::io::{Cursor, Read, Write};
/// use webagents::model::accel::generate;
///
/// /// A scripted accelerator: it records what it is sent and plays an answer back.
/// struct Scripted {
///     answer: Cursor<&'static [u8]>,
///     sent: Vec<u8>,
/// }
/// impl Read for Scripted {
///     fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
///         self.answer.read(buf)
///     }
/// }
/// impl Write for Scripted {
///     fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
///         self.sent.write(buf)
///     }
///     fn flush(&mut self) -> std::io::Result<()> {
///         Ok(())
///     }
/// }
///
/// // Two tokens, then the stop id 2, then "done after 3 tokens, by a stop".
/// let mut link = Scripted { answer: Cursor::new(&b"T 11\nT 12\nT 2\nD eos 3\n"[..]), sent: Vec::new() };
/// let generation = generate(&mut link, &[5, 6, 7], 16, &[2], |_id| {}).unwrap();
///
/// assert_eq!(generation.tokens, [11, 12, 2]);
/// assert!(generation.stopped);
/// // What went to the accelerator: reset, the prompt, the generate request.
/// assert_eq!(link.sent, b"R\nP 3 5 6 7\nG 16 2\n");
/// ```
pub fn generate<S: io::Read + Write>(
    link: &mut S,
    prompt: &[u32],
    max_new: u32,
    stops: &[u32],
    mut on_token: impl FnMut(u32),
) -> Result<Generation, LinkError> {
    if prompt.is_empty() || stops.is_empty() {
        return Err(LinkError::Protocol("an empty prompt or no stop id".into()));
    }
    let mut out = String::from("R\n");
    for chunk in prompt.chunks(PROMPT_MAX) {
        out.push_str(&format!("P {}", chunk.len()));
        for id in chunk {
            out.push_str(&format!(" {id}"));
        }
        out.push('\n');
    }
    out.push_str(&format!("G {max_new} {}\n", stops.iter().map(u32::to_string).collect::<Vec<_>>().join(",")));
    link.write_all(out.as_bytes())?;
    link.flush()?;
    let mut r = BufReader::new(link);
    let mut tokens = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return Err(LinkError::Io(io::Error::new(io::ErrorKind::UnexpectedEof, "the link closed")));
        }
        let l = line.trim_end();
        if let Some(t) = l.strip_prefix("T ") {
            let id = t.parse().map_err(|_| LinkError::Protocol(l.to_string()))?;
            on_token(id);
            tokens.push(id);
            if tokens.len() > max_new as usize {
                return Err(LinkError::Protocol(format!("more than {max_new} tokens")));
            }
        } else if let Some(d) = l.strip_prefix("D ") {
            let mut it = d.split(' ');
            let stopped = match it.next() {
                Some("eos") => true,
                Some("max") => false,
                _ => return Err(LinkError::Protocol(l.to_string())),
            };
            let n: usize = it.next().and_then(|n| n.parse().ok()).ok_or_else(|| LinkError::Protocol(l.to_string()))?;
            if n != tokens.len() {
                return Err(LinkError::Protocol(format!("done after {n} tokens, {} received", tokens.len())));
            }
            return Ok(Generation { tokens, stopped });
        } else if let Some(e) = l.strip_prefix("E") {
            return Err(LinkError::Refused(e.trim().to_string()));
        } else {
            return Err(LinkError::Protocol(l.to_string()));
        }
    }
}

/// Emits one generated token on the link.
pub type Emit<'a> = dyn FnMut(u32) -> io::Result<()> + 'a;

/// The model behind [`serve_one`]: `(prompt, max_new, stops, emit)`, returning
/// whether a stop id ended the run.
pub type ChipModel<'a> = dyn FnMut(&[u32], u32, &[u32], &mut Emit<'_>) -> io::Result<bool> + 'a;

/// The chip side, for twins and tests: reads one request and answers it with
/// `model(prompt, max_new, stops, emit)`, which calls `emit` per token and
/// returns whether a stop ended the run. Returns false when the link closed.
pub fn serve_one<S: io::Read + Write>(link: &mut S, model: &mut ChipModel<'_>) -> io::Result<bool> {
    let mut prompt: Vec<u32> = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        let n = {
            let mut byte = [0u8; 1];
            let mut got = 0;
            loop {
                match link.read(&mut byte)? {
                    0 => break,
                    _ => {
                        got += 1;
                        if byte[0] == b'\n' {
                            break;
                        }
                        line.push(byte[0] as char);
                    }
                }
            }
            got
        };
        if n == 0 {
            return Ok(false);
        }
        let mut it = line.trim_end().split(' ');
        match it.next() {
            Some("R") => prompt.clear(),
            Some("P") => {
                let k: usize = it.next().and_then(|k| k.parse().ok()).unwrap_or(usize::MAX);
                let ids: Vec<u32> = it.filter_map(|x| x.parse().ok()).collect();
                if k > PROMPT_MAX || ids.len() != k {
                    link.write_all(format!("E a prompt event carries 1 to {PROMPT_MAX} ids\n").as_bytes())?;
                    return Ok(true);
                }
                prompt.extend(ids);
            }
            Some("G") => {
                let max_new: u32 = it.next().and_then(|m| m.parse().ok()).unwrap_or(0);
                let stops: Vec<u32> = it.next().unwrap_or("").split(',').filter_map(|s| s.parse().ok()).collect();
                if prompt.is_empty() || stops.is_empty() || max_new == 0 {
                    link.write_all(b"E nothing to generate\n")?;
                    return Ok(true);
                }
                let mut count = 0usize;
                let stopped = {
                    let mut emit = |id: u32| -> io::Result<()> {
                        count += 1;
                        link.write_all(format!("T {id}\n").as_bytes())?;
                        link.flush()
                    };
                    model(&prompt, max_new, &stops, &mut emit)?
                };
                link.write_all(format!("D {} {count}\n", if stopped { "eos" } else { "max" }).as_bytes())?;
                link.flush()?;
                return Ok(true);
            }
            _ => {
                link.write_all(b"E unknown request\n")?;
                return Ok(true);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};

    #[test]
    fn long_prompts_are_cut_to_the_port_and_tokens_stream_back() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let chip = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut seen = Vec::new();
            serve_one(&mut s, &mut |p, max_new, stops, emit| {
                seen = p.to_vec();
                // A toy "model": echo the last three prompt ids, then the stop.
                for id in p.iter().rev().take(3) {
                    emit(*id)?;
                }
                emit(stops[0])?;
                assert!(max_new >= 4);
                Ok(true)
            })
            .unwrap();
            seen
        });
        let mut s = TcpStream::connect(addr).unwrap();
        let prompt: Vec<u32> = (0..150).collect();
        let mut streamed = Vec::new();
        let g = generate(&mut s, &prompt, 16, &[7, 9], |t| streamed.push(t)).unwrap();
        assert_eq!(g, Generation { tokens: vec![149, 148, 147, 7], stopped: true });
        assert_eq!(streamed, g.tokens);
        assert_eq!(chip.join().unwrap(), prompt, "150 ids arrive whole, in order, over 3 prompt events");
    }
}
