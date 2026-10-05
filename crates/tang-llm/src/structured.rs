//! Local JSON/schema token constraints, with bounded request compilation.
use anyhow::{bail, ensure, Result};
use llguidance::{
    api::TopLevelGrammar,
    toktrie::{ApproximateTokEnv, TokRxInfo, TokTrie},
    JsonCompileOptions, Matcher, ParserFactory,
};
use serde_json::{json, Value};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Arc, LazyLock},
};

const MAX_SCHEMA_BYTES: usize = 64 * 1024;
static VALIDATOR: LazyLock<ParserFactory> = LazyLock::new(|| {
    let mut factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env())
        .expect("byte tokenizer factory");
    factory.quiet();
    factory
});

fn grammar(mut schema: Value) -> TopLevelGrammar {
    if schema.is_boolean() {
        schema = json!({"allOf":[schema]});
    }
    // A skip lexeme is allowed once between grammar terminals. Bound it so a
    // model cannot spend its entire response repeating indentation before a value.
    // Whitespace inside JSON strings remains part of their content, unrestricted.
    JsonCompileOptions {
        whitespace_pattern: Some(r"[ \t\r\n]{1,8}".into()),
        ..Default::default()
    }
    .apply_to(&mut schema);
    TopLevelGrammar::from_json_schema(schema)
}

pub fn response_schema(body: &Value) -> Result<Option<Value>> {
    let format = &body["response_format"];
    if format.is_null() {
        return Ok(None);
    }
    let schema = match format["type"].as_str() {
        Some("text") => return Ok(None),
        Some("json_object") => json!({"type":"object"}),
        Some("json_schema") => format["json_schema"]
            .get("schema")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("response_format.json_schema.schema is required"))?,
        _ => bail!("unsupported response_format type"),
    };
    ensure!(
        schema.is_object() || schema.is_boolean(),
        "schema must be an object or boolean"
    );
    ensure!(
        serde_json::to_vec(&schema)?.len() <= MAX_SCHEMA_BYTES,
        "schema exceeds 64 KiB"
    );
    ensure!(
        body["stop"].is_null() || body["stop"].as_array().is_some_and(Vec::is_empty),
        "stop strings cannot be combined with structured output"
    );
    // Validate without a model or network access, before this request enters the queue.
    catch_unwind(AssertUnwindSafe(|| {
        VALIDATOR.create_parser(grammar(schema.clone()))
    }))
    .map_err(|_| anyhow::anyhow!("invalid structured output schema"))??;
    Ok(Some(schema))
}

pub fn model_factory(
    tokenizer: &tokenizers::Tokenizer,
    vocab: usize,
    eos: u32,
) -> Result<ParserFactory> {
    let json: Value = serde_json::from_str(
        &tokenizer
            .to_string(false)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?,
    )?;
    let mut bytes = catch_unwind(AssertUnwindSafe(|| {
        llguidance::token_bytes_from_tokenizer_json(&json)
    }))
    .map_err(|_| anyhow::anyhow!("unsupported structured-output tokenizer"))??;
    ensure!(
        bytes.len() <= vocab && (eos as usize) < vocab,
        "tokenizer vocabulary differs from model"
    );
    bytes.resize(vocab, Vec::new());
    let env = Arc::new(ApproximateTokEnv::new(TokTrie::from(
        &TokRxInfo::new(vocab as u32, eos),
        &bytes,
    )));
    let mut factory = ParserFactory::new_simple(&(env as llguidance::toktrie::TokEnv))?;
    factory.quiet();
    Ok(factory)
}

pub fn matcher(factory: &ParserFactory, schema: Value) -> Result<Matcher> {
    ensure!(
        serde_json::to_vec(&schema)?.len() <= MAX_SCHEMA_BYTES,
        "schema exceeds 64 KiB"
    );
    let parser = factory.create_parser(grammar(schema))?;
    Ok(Matcher::new(Ok(parser)))
}

pub fn mask(matcher: &mut Matcher, logits: &mut [f32], eos: &[u32]) -> Result<()> {
    let accepting = matcher.is_accepting()?;
    let allowed = matcher.compute_mask_or_eos()?;
    let mut any = false;
    for (token, logit) in logits.iter_mut().enumerate() {
        if (eos.contains(&(token as u32)) && accepting)
            || (!eos.contains(&(token as u32)) && allowed.is_allowed(token as u32))
        {
            any |= logit.is_finite();
        } else {
            *logit = f32::NEG_INFINITY;
        }
    }
    ensure!(any, "structured output has no finite allowed token");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_formats_are_bounded_and_validated() {
        assert!(response_schema(&json!({})).unwrap().is_none());
        assert!(response_schema(&json!({"response_format":{"type":"text"}}))
            .unwrap()
            .is_none());
        assert!(
            response_schema(&json!({"response_format":{"type":"json_object"}}))
                .unwrap()
                .is_some()
        );
        for format in [
            json!({"type":"bad"}),
            json!({"type":"json_schema"}),
            json!({"type":"json_schema","json_schema":{"schema":{"type":"bogus"}}}),
        ] {
            assert!(response_schema(&json!({"response_format":format})).is_err());
        }
        assert!(
            response_schema(&json!({"stop":"}","response_format":{"type":"json_object"}})).is_err()
        );
        assert!(response_schema(&json!({"response_format":{"type":"json_schema","json_schema":{"schema":{"description":"x".repeat(MAX_SCHEMA_BYTES)}}}})).is_err());
    }
    #[test]
    fn schema_masks_bad_keys_values_and_premature_eos() {
        let schema = json!({"type":"object","properties":{"tool":{"const":"Write"},"args":{"type":"object","properties":{"content":{"type":"string"}},"required":["content"],"additionalProperties":false}},"required":["tool","args"],"additionalProperties":false});
        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let valid = b"{\"tool\":\"Write\",\"args\":{\"content\":\"<div>frog</div>\\n\"}}";
        let mut grammar = matcher(&factory, schema.clone()).unwrap();
        for &byte in valid {
            let mut logits = vec![0.; 257];
            mask(&mut grammar, &mut logits, &[256]).unwrap();
            assert!(logits[byte as usize].is_finite());
            grammar.consume_token(byte as u32).unwrap();
        }
        assert!(grammar.is_accepting().unwrap());
        let mut logits = vec![0.; 257];
        mask(&mut grammar, &mut logits, &[255, 256]).unwrap();
        assert!(logits[255].is_finite() && logits[256].is_finite());
        for invalid in [
            b"{\"tool\":\"Bash\"}".as_slice(),
            b"{\"args\":{\"bad\":1}}".as_slice(),
            b"{\"tool\"<".as_slice(),
        ] {
            let mut grammar = matcher(&factory, schema.clone()).unwrap();
            let mut rejected = false;
            for &byte in invalid {
                let allowed = grammar.compute_mask().unwrap();
                if !allowed.is_allowed(byte as u32) {
                    rejected = true;
                    break;
                }
                grammar.consume_token(byte as u32).unwrap();
            }
            assert!(rejected);
        }
    }

    #[test]
    fn accepting_number_can_continue_and_eos_is_blocked_before_acceptance() {
        let factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
        let mut number = matcher(&factory, json!({"type":"integer"})).unwrap();
        number.consume_token(b'1' as u32).unwrap();
        let mut logits = vec![0.; 257];
        mask(&mut number, &mut logits, &[256]).unwrap();
        assert!(logits[b'0' as usize].is_finite() && logits[256].is_finite());
        let mut object = matcher(&factory, json!({"type":"object"})).unwrap();
        object.consume_token(b'{' as u32).unwrap();
        logits.fill(0.);
        mask(&mut object, &mut logits, &[255, 256]).unwrap();
        assert!(!logits[255].is_finite() && !logits[256].is_finite());
    }

    #[test]
    fn indentation_cannot_loop_but_string_whitespace_is_preserved() {
        let factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
        let mut object = matcher(&factory, json!({"type":"object","properties":{"count":{"const":7}},"required":["count"],"additionalProperties":false})).unwrap();
        for &byte in b"{\"count\":" {
            object.consume_token(byte as u32).unwrap();
        }
        for _ in 0..8 {
            let allowed = object.compute_mask().unwrap();
            assert!(allowed.is_allowed(b' ' as u32));
            object.consume_token(b' ' as u32).unwrap();
        }
        let allowed = object.compute_mask().unwrap();
        assert!(!allowed.is_allowed(b' ' as u32));
        assert!(allowed.is_allowed(b'7' as u32));
        let mut string = matcher(&factory, json!({"type":"string"})).unwrap();
        for &byte in serde_json::to_vec(&format!("{}\n", " ".repeat(30)))
            .unwrap()
            .iter()
        {
            let allowed = string.compute_mask().unwrap();
            assert!(allowed.is_allowed(byte as u32));
            string.consume_token(byte as u32).unwrap();
        }
        assert!(string.is_accepting().unwrap());
        assert!(response_schema(
            &json!({"response_format":{"type":"json_schema","json_schema":{"schema":true}}})
        )
        .unwrap()
        .is_some());
    }
}
