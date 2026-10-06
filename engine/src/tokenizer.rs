use crate::gguf::Gguf;
use anyhow::{ensure, Context, Result};
use std::collections::{BinaryHeap, HashMap};

pub struct Tokenizer {
    pub tokens: Vec<String>,
    types: Vec<u32>,
    ids: HashMap<String, u32>,
    merges: HashMap<(String, String), u32>,
    specials: Vec<(String, u32)>,
    bytes: [Option<u32>; 256],
    pub template: String,
    /// GPT-2 style byte-level BPE (Qwen); `None` is Gemma's SentencePiece-style BPE.
    byte_level: Option<ByteLevel>,
}
/// Byte-to-character table and pre-tokenizer of a byte-level BPE vocabulary.
struct ByteLevel {
    encode: [char; 256],
    decode: HashMap<char, u8>,
    split: fancy_regex::Regex,
    bos: Option<u32>,
}
impl ByteLevel {
    fn new(g: &Gguf) -> Result<Self> {
        // llama.cpp's expansion of the tokenizer.json pattern, without the inline (?i:) group.
        let pattern = match g.get_str("tokenizer.ggml.pre")? {
            "qwen35" => r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            other => anyhow::bail!("unsupported byte-level pre-tokenizer {other}"),
        };
        let mut encode = ['\0'; 256];
        let mut extra = 0;
        for (b, c) in encode.iter_mut().enumerate() {
            let printable = matches!(b, 33..=126 | 161..=172 | 174..=255);
            *c = char::from_u32(if printable { b as u32 } else { 256 + extra }).unwrap();
            extra += u32::from(!printable);
        }
        let add_bos = g
            .get("tokenizer.ggml.add_bos_token")
            .ok()
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            != 0;
        Ok(Self {
            decode: encode.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect(),
            encode,
            split: fancy_regex::Regex::new(pattern)?,
            bos: if add_bos {
                Some(g.get_u64("tokenizer.ggml.bos_token_id")? as u32)
            } else {
                None
            },
        })
    }
}
impl Tokenizer {
    pub fn new(g: &Gguf) -> Result<Self> {
        let byte_level = match g.get_str("tokenizer.ggml.model")? {
            "gemma4" => None,
            "gpt2" => Some(ByteLevel::new(g)?),
            other => anyhow::bail!("unsupported tokenizer {other}"),
        };
        let tokens: Vec<_> = g
            .get("tokenizer.ggml.tokens")?
            .as_arr()
            .context("tokenizer metadata must be an array")?
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_owned)
                    .context("non-string tokenizer token")
            })
            .collect::<Result<Vec<_>>>()?;
        let types: Vec<_> = g
            .get("tokenizer.ggml.token_type")?
            .as_arr()
            .context("tokenizer metadata must be an array")?
            .iter()
            .map(|x| {
                x.as_u64()
                    .context("non-integer token type")
                    .and_then(|n| u32::try_from(n).map_err(Into::into))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(tokens.len() == types.len(), "token type count mismatch");
        let ids: HashMap<_, _> = tokens
            .iter()
            .enumerate()
            .map(|(i, x)| (x.clone(), i as u32))
            .collect();
        let mut merges = HashMap::new();
        for (i, m) in g
            .get("tokenizer.ggml.merges")?
            .as_arr()
            .context("tokenizer metadata must be an array")?
            .iter()
            .enumerate()
        {
            let (a, b) = m
                .as_str()
                .context("non-string merge")?
                .split_once(' ')
                .context("invalid BPE merge")?;
            merges.insert((a.to_owned(), b.to_owned()), i as u32);
        }
        let mut specials: Vec<_> = tokens
            .iter()
            .zip(&types)
            .enumerate()
            .filter(|(_, (s, t))| matches!(t, 2..=4) && s.as_str() != "</s>")
            .map(|(i, (s, _))| (s.clone(), i as u32))
            .collect();
        specials.sort_by_key(|(s, id)| (std::cmp::Reverse(s.len()), *id));
        let mut bytes = [None; 256];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = match &byte_level {
                Some(bl) => ids.get(&bl.encode[i].to_string()),
                None => ids.get(&format!("<0x{i:02X}>")),
            }
            .copied();
        }
        ensure!(
            bytes.iter().all(Option::is_some),
            "missing byte fallback token"
        );
        Ok(Self {
            tokens,
            types,
            ids,
            merges,
            specials,
            bytes,
            template: g.get_str("tokenizer.chat_template")?.into(),
            byte_level,
        })
    }
    pub fn id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }
    pub fn encode(&self, text: &str, bos: bool, special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        match &self.byte_level {
            Some(bl) => out.extend(bl.bos.filter(|_| bos)),
            None if bos => out.push(2),
            None => {}
        }
        let mut pos = 0;
        let mut plain = 0;
        while pos < text.len() {
            if let Some((s, id)) = if special {
                self.specials
                    .iter()
                    .find(|(s, _)| text[pos..].starts_with(s))
            } else {
                None
            } {
                self.plain(&text[plain..pos], &mut out);
                out.push(*id);
                pos += s.len();
                plain = pos;
            } else {
                pos += text[pos..].chars().next().unwrap().len_utf8();
            }
        }
        self.plain(&text[plain..], &mut out);
        out
    }
    fn plain(&self, text: &str, out: &mut Vec<u32>) {
        if let Some(bl) = &self.byte_level {
            for word in bl.split.find_iter(text) {
                let word = word.expect("pre-tokenizer backtracking limit");
                let mapped: String = word.as_str().bytes().map(|b| bl.encode[b as usize]).collect();
                self.bpe(&mapped, out);
            }
            return;
        }
        let text = text.replace(' ', "▁");
        let mut start = 0;
        while start < text.len() {
            let nl = text.as_bytes()[start] == b'\n';
            let end = text[start..]
                .char_indices()
                .find(|&(_, c)| (c == '\n') != nl)
                .map(|(i, _)| start + i)
                .unwrap_or(text.len());
            let run = &text[start..end];
            if nl {
                if let Some(id) = self.ids.get(run) {
                    out.push(*id);
                    start = end;
                    continue;
                }
            }
            self.bpe(run, out);
            start = end;
        }
    }
    fn bpe(&self, text: &str, out: &mut Vec<u32>) {
        let mut pieces: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let n = pieces.len();
        if n == 0 {
            return;
        }
        let mut next: Vec<usize> = (1..=n).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut gen = vec![0usize; n];
        let mut queue = BinaryHeap::new();
        let enqueue = |i: usize,
                       pieces: &Vec<String>,
                       next: &Vec<usize>,
                       gen: &Vec<usize>,
                       queue: &mut BinaryHeap<_>| {
            if i < n && next[i] < n {
                if let Some(rank) = self
                    .merges
                    .get(&(pieces[i].clone(), pieces[next[i]].clone()))
                {
                    queue.push(std::cmp::Reverse((*rank, i, gen[i], next[i], gen[next[i]])));
                }
            }
        };
        for i in 0..n {
            enqueue(i, &pieces, &next, &gen, &mut queue);
        }
        while let Some(std::cmp::Reverse((_, i, gi, j, gj))) = queue.pop() {
            if gen[i] != gi || gen[j] != gj || next[i] != j {
                continue;
            }
            let right = std::mem::take(&mut pieces[j]);
            pieces[i].push_str(&right);
            gen[i] += 1;
            gen[j] += 1;
            next[i] = next[j];
            if next[j] < n {
                prev[next[j]] = i;
            }
            enqueue(prev[i], &pieces, &next, &gen, &mut queue);
            enqueue(i, &pieces, &next, &gen, &mut queue);
        }
        for piece in pieces.into_iter().filter(|s| !s.is_empty()) {
            if let Some(id) = self.ids.get(&piece) {
                out.push(*id);
            } else if let Some(bl) = &self.byte_level {
                out.extend(piece.chars().filter_map(|c| self.bytes[bl.decode[&c] as usize]));
            } else {
                for b in piece.bytes() {
                    if let Some(id) = self.bytes[b as usize] {
                        out.push(id);
                    }
                }
            }
        }
    }
    pub fn decode_bytes(&self, id: u32, special: bool) -> Vec<u8> {
        let Some(s) = self.tokens.get(id as usize) else {
            return vec![];
        };
        if let Some(bl) = &self.byte_level {
            return match self.types[id as usize] {
                1 => s.chars().filter_map(|c| bl.decode.get(&c).copied()).collect(),
                3 if !special => vec![],
                _ => s.clone().into_bytes(),
            };
        }
        if self.types[id as usize] == 6 && s.len() == 6 {
            if let Ok(b) = u8::from_str_radix(&s[3..5], 16) {
                return vec![b];
            }
        }
        if !special && matches!(self.types[id as usize], 2..=4) {
            return vec![];
        }
        s.replace('▁', " ").into_bytes()
    }
    pub fn decode(&self, ids: &[u32], special: bool) -> String {
        String::from_utf8_lossy(
            &ids.iter()
                .flat_map(|id| self.decode_bytes(*id, special))
                .collect::<Vec<_>>(),
        )
        .into_owned()
    }
    pub fn render(&self, messages: &[serde_json::Value], thinking: bool) -> Result<String> {
        self.render_with_tools(messages, thinking, &[])
    }
    pub fn render_with_tools(
        &self,
        messages: &[serde_json::Value],
        thinking: bool,
        tools: &[serde_json::Value],
    ) -> Result<String> {
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.add_template("chat", &self.template)?;
        Ok(env.get_template("chat")?.render(minijinja::context!{messages=>messages,add_generation_prompt=>true,enable_thinking=>thinking,bos_token=>"",eos_token=>"<eos>",tools=>tools})?)
    }
    pub fn fixture(&self, path: &std::path::Path) -> Result<()> {
        let cases: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(path)?)?;
        for c in &cases {
            let text = c["text"].as_str().unwrap();
            let ids = self.encode(text, true, true);
            let want: Vec<u32> = serde_json::from_value(c["ids"].clone())?;
            ensure!(
                ids == want,
                "tokenizer mismatch {text:?}: {ids:?} != {want:?}"
            );
            ensure!(
                self.decode(&ids, true) == c["detokenized"].as_str().unwrap(),
                "decode mismatch {text:?}"
            );
        }
        println!("{} tokenizer fixtures passed", cases.len());
        Ok(())
    }
    pub fn chat_fixture(&self, path: &std::path::Path) -> Result<()> {
        let cases: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(path)?)?;
        for c in &cases {
            let messages = c["messages"]
                .as_array()
                .context("missing fixture messages")?;
            let thinking = c["thinking"].as_bool().context("missing thinking flag")?;
            let rendered = self.render(messages, thinking)?;
            ensure!(
                Some(rendered.as_str()) == c["rendered"].as_str(),
                "chat rendering mismatch: {}",
                c["name"]
            );
            if let Some(ids) = c.get("ids") {
                let want: Vec<u32> = serde_json::from_value(ids.clone())?;
                ensure!(
                    self.encode(&rendered, true, true) == want,
                    "chat token mismatch: {}",
                    c["name"]
                );
            }
        }
        println!("{} chat fixtures passed", cases.len());
        Ok(())
    }
}
