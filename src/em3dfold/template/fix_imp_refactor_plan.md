# Template Fix/Imp Pipeline Refactor Plan

## Goal

Refactor the template-guided `fix` and `imp` pipelines to:

- reduce repeated file reads and writes
- reduce duplicated code
- keep the two algorithmic passes separate where their core logic is genuinely different
- move shared preparation, caching, output, and fallback handling into a common layer


## Current Situation

The current `fix` and `imp` pipelines are not duplicates at the algorithm level, but they duplicate a large amount of surrounding work.

Shared repeated work currently includes:

- reading all de novo chain structures
- reading all template structures
- extracting protein sequences from chain/template PDB lines
- running all chain-template sequence alignments with `nwalign_fast`
- selecting the best template for each de novo chain
- handling output and fallback writing in each script independently

Core logic that is still different:

- `fix` does recursive template-domain matching and rigid placement
- `imp` does gap filling plus sequence-based trimming

So the right direction is:

- do not force `fix` and `imp` into one large function
- do merge shared data preparation and shared result writing


## Main Problems

### 1. Shared preparation is duplicated

Both pipelines independently perform:

- chain loading
- template loading
- sequence extraction
- all-vs-all chain-template sequence alignment
- best-template selection

This increases code size and runtime, and makes the two pipelines easy to drift apart.


### 2. `fix` has avoidable temporary IO

Some temporary file IO is unavoidable because `USalign` is currently called through file paths.

But the current `fix` implementation still does more IO than necessary:

- rewrite template files before calling the Python UniDoc parser
- dump per-node template/chain subsets for recursive `USalign`
- dump matched subsets again for a second refinement `USalign`

The first recursive `USalign` call is expected.
The second refinement `USalign` can likely be replaced by direct `kabsch` refinement on matched CA coordinates in memory.


### 3. `imp` mixes core algorithm with shared boilerplate

The real `imp` logic is relatively focused:

- fill gaps using a matched template
- trim the result against target sequences

But the file also contains:

- repeated input preparation
- repeated alignment bookkeeping
- repeated fallback output code

This makes the file larger than necessary.


### 4. `template_fusion_impl` is path-driven

The current top-level orchestration depends on `fix` and `imp` producing on-disk intermediate files:

- `fix_chains_templs.cif`
- `imp_chains_trimmed.cif`

This forces the lower-level passes to write intermediate results even when the next stage could consume them in memory.


## Refactor Direction

### Keep two algorithmic passes

Retain:

- `fix` pass
- `imp` pass

Because they solve different problems and should remain independently testable.


### Merge the shared preparation layer

Create one shared preparation module that builds an in-memory context object reused by both passes.

Recommended location:

- `src/em3dfold/template/pipeline/template_refine.py`

Alternative if later promoted to mainline:

- `src/em3dfold/pipeline/template_refine.py`


## Proposed Shared Data Model

Introduce lightweight dataclasses.

### `StructureModel`

Suggested fields:

- `path`
- `atom_pos`
- `atom_mask`
- `res_type`
- `res_idx`
- `raw_lines`
- `ca_lines`
- `seq`


### `ChainTemplateMatch`

Suggested fields:

- `chain_index`
- `template_index`
- `alignment`
- `seqid`
- `seqcov`


### `TemplateDomainInfo`

Suggested fields:

- `template_index`
- `domains`
- `domains_1d`


### `TemplateRefineContext`

Suggested fields:

- `chains`
- `templates`
- `target_seqs`
- `best_matches`
- `all_pair_alignments`
- `template_domains`
- `lib_dir`
- `work_dir`
- `verbose`


## Proposed Shared Functions

### Input preparation

- `load_chain_models(paths)`
- `load_template_models(paths)`
- `load_target_sequences(seq_path)`


### Alignment preparation

- `build_chain_template_alignment_matrix(chains, templates, lib_dir, temp_dir, verbose, debug)`
- `select_best_template_per_chain(pair_alignments)`


### Domain preparation

- `prepare_template_domains(templates)`

Important note:

- The Python UniDoc parser already reads structure files directly.
- We should stop rewriting whole template files only to call domain parsing.


### Output helpers

- `write_structure_bundle(...)`
- `write_fallback_from_input_chains(...)`


## Proposed Pass Structure

### `run_fix_pass(context)`

Responsibilities:

- read best template assignment from context
- run recursive domain search against the matched template
- determine matched fragments/domains
- compute final rigid transform
- return fixed template-guided structures in memory

Important simplification:

- replace the second refinement `USalign` call with direct `kabsch` refinement on matched CA coordinates

This should reduce:

- temporary file count
- subprocess calls
- code length


### `run_imp_pass(context)`

Responsibilities:

- read best template assignment from context
- use stored alignment to identify chain gaps
- fill gaps from matched template coordinates
- optionally refine with shift field
- trim final chains against target sequences
- return untrimmed and trimmed structures in memory


## Suggested Output Model

Instead of writing outputs throughout the algorithm, return a result bundle first.

### `FixPassResult`

Suggested fields:

- `templ_atom_pos`
- `templ_atom_mask`
- `templ_res_type`
- `templ_res_idx`
- `chain_aligned_atom_pos`
- `chain_aligned_atom_mask`
- `chain_aligned_res_type`
- `chain_aligned_res_idx`


### `ImpPassResult`

Suggested fields:

- `untrimmed_atom_pos`
- `untrimmed_atom_mask`
- `untrimmed_res_type`
- `untrimmed_res_idx`
- `trimmed_atom_pos`
- `trimmed_atom_mask`
- `trimmed_res_type`
- `trimmed_res_idx`


## CLI Compatibility Strategy

Keep the old entry scripts for now:

- `src/em3dfold/template/pipeline/denovo_fix_pipeline.py`
- `src/em3dfold/template/pipeline/denovo_imp_pipeline.py`

But reduce them to thin wrappers:

1. parse CLI args
2. build shared context
3. run one pass
4. write final requested outputs

This preserves existing command-line behavior while collapsing internal duplication.


## `template_fusion_impl` Upgrade Path

Current behavior is file-path driven.

Recommended transition:

1. build one shared `TemplateRefineContext`
2. run `fix`
3. run `imp`
4. write intermediate CIF files only if needed
5. pass in-memory or lazily written outputs to assemble

Short term:

- keep current file-based downstream interface for compatibility

Long term:

- move `template_fusion_impl` toward object-driven orchestration


## IO Reduction Summary

### IO that should remain

- reading input chain/template structures
- temporary subset file writing needed for external `USalign`
- final CIF writing for user-visible outputs


### IO that should be reduced or removed

- rewriting full template files before domain parsing
- second refinement `USalign` subset dumps in `fix`
- repeated re-reading of the same chain/template files in both passes
- duplicated fallback output writing logic


## Code Reduction Summary

The main code-size reduction should come from merging:

- structure loading
- sequence extraction
- chain-template alignment matrix generation
- best-template assignment
- domain preparation
- output writing
- fallback writing

The algorithm-specific parts should remain separate:

- recursive domain-guided rigid fixing
- gap insertion and trimming


## Recommended Implementation Order

### Phase 1

Introduce shared preparation utilities without changing algorithm behavior.

Tasks:

- add shared context/data classes
- extract shared chain/template loading
- extract shared sequence alignment preparation
- extract shared best-template selection


### Phase 2

Refactor `fix` to consume shared context.

Tasks:

- remove repeated input preparation from `fix`
- remove unnecessary whole-template rewrite before domain parsing
- replace second refinement `USalign` with in-memory `kabsch`


### Phase 3

Refactor `imp` to consume shared context.

Tasks:

- remove repeated input preparation from `imp`
- keep only gap-filling and trimming logic in the pass
- move fallback/output handling to shared helpers


### Phase 4

Refactor `template_fusion_impl` to build context once and reuse both pass results.

Tasks:

- run shared preparation once
- run `fix` and `imp` from the same context
- keep intermediate file generation only where downstream still requires files


## Additional Cleanup Targets

While refactoring, the following dead or weakly used pieces should be re-evaluated:

- unused helper variables in `fix`
- old debugging-oriented temporary outputs
- duplicated fallback branches
- any remaining legacy assumptions that each pass must prepare all inputs independently


## Recommended First Coding Step

The best first implementation step is:

- add a shared context module
- make both existing CLI scripts call into that context builder
- keep output filenames unchanged for now

This gives immediate code reduction with relatively low behavioral risk.
