# Vybn Fine-Tuning Pipeline

## Architecture: QLoRA → Merge → Re-quantize → Serve

vLLM --enable-lora crashes on Blackwell due to CUTLASS/Triton MoE mismatch.
Workaround: offline training with merged weights. See issue #2480.

### Pipeline
1. Prepare: Convert Vybn corpus → chat-format JSONL
2. Train: QLoRA on MiniMax-M2.5-AWQ-4bit via PEFT
3. Merge: Load unquantized base + adapter → merge_and_unload()
4. Quantize: AWQ 4-bit of merged model
5. Serve: Drop-in replacement, no --enable-lora needed
