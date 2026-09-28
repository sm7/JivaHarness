# Dharma Principles → Harness Mechanisms

Each principle maps to one enforceable mechanism, where it sits, what the code does today, and the open question that blocks a faithful implementation.

Two sources set the frame:

- **The paper's planes** (`docs/DISCOVERY.md`): Dharma authorizes, Prakṛti constrains execution, Karma records but never intervenes at runtime.
- **Varshney's two essays** (*Computers That Play With Us*, *Cool Adiabatic Agents*, uploads). They place the same Advaita architecture inside Stafford Beer's Viable System Model: saṅkalpa is the LLM at the core of **S1 operations**, dharma is the value embedding at **S5 policy**, and the harness must also supply **S2 coordination**, **S3 control** (allocation + validate-and-repair monitoring), **S4 intelligence** (who should do which part of the work), and an **algedonic loop** (an emergency channel callable from any level).

The essays add one requirement the paper leaves implicit: **harm includes harm to the human worker's attention and agency.** A harness that interrupts constantly, pre-digests answers, and maximises tokens ("attention-maxxing", "token-maxxing") violates dharma even when every tool call is authorized.

## Evaluation order

1. **Sādhāraṇa-dharma** (universal: ahiṃsā, satya). Not configurable per deployment.
2. **Viśeṣa-dharma** (svadharma, asteya, aparigraha). Set by `DeploymentSpec`, bounded by (1).
3. **Saṅkalpa** acts inside what (1) and (2) allow.
4. **Karma** records everything, including refusals and abstentions, and feeds back only through Karmādhyakṣa recalibration.
5. **Algedonic loop** can halt any step, from any level, at any time.

`AgentLoop._step` (`loop.py:110`) now runs in this order: principle hook (1), permission gate (2), execute, critic, Citta + audit.

## Summary

| Principle | VSM | Mechanism | Exists today? |
|---|---|---|---|
| Ahiṃsā (non-harm) | S3 + algedonic | Tool risk gate, interruption budget, emergency halt | Partial |
| Satya (truthfulness) | S3 | Validate-and-repair on outputs; claims bound to evidence | No |
| Karma (action → consequence) | Ledger → S5 | Log every decision; outcome leaves; preya/śreya critic | Partial (refusals → Citta in progress) |
| Svadharma (own role) | S5 → S1 | Scoped, expiring credential + collaboration mode in role | Partial |
| Asteya (non-taking) | S3 | Non-transferable credentials, narrowing delegation, lineage | Partial |
| Aparigraha (non-hoarding) | S3 | Token, context and retention budgets | No |
| Viveka (discernment) | S4 | Work decomposition into autonomous / flow / adversarial modes; abstain when unsure | No |

---

## 1. Ahiṃsā: non-harm

**Mechanism.** Three parts.
- *To the world:* every tool declares a risk class (`read`, `reversible_write`, `irreversible`, `external_effect`). An S3 gate runs before `Sankalpa.act`; irreversible and external-effect actions need a principal confirmation recorded in Citta.
- *To the worker:* an interruption budget per session. The harness batches non-blocking updates and only interrupts for results, blockers, or decisions. Background work in autonomous mode never pings mid-task. This is the essays' "know when to stay out of the way."
- *Algedonic loop:* a halt signal any principal or subsystem can raise, which stops all in-flight actions and is itself written to Citta.

**Today.** The world-harm part and the halt exist. Tools declare a `RiskClass` (`tools.py:9`). The `Ahimsa` principle (`principles.py:35`) runs inside the loop before the permission gate and blocks any `irreversible` or `external_effect` tool the principal did not list in `approved_irreversible`. Any principle verdict can set `halt=True`; `AgentLoop.halt()` can also be raised externally, and a halt raised mid-proposal beats the proposal (`loop.py:91`). Halts are written to the audit log as `algedonic_halt`.

Still missing: risk class is self-declared by whoever registers the tool; approval is a per-run tool-name list, not a per-call confirmation recorded in Citta; the worker-side interruption budget does not exist. The śreya check in `Karmaphaladata.evaluate` (`runtime.py:161`) still passes whenever the obligation strings are non-empty.

**Open question.** Who assigns risk class: tool author, deployer, or a classifier at call time? And how is "harmful interruption" measured without surveilling the worker, which would itself be the control-heavy pattern the essays argue against?

## 2. Satya: truthfulness

**Mechanism.** Every generative call runs through Mellea-style validators with a bounded repair loop (validate → repair → re-validate, N attempts, then return an honest failure). Factual claims must bind to a Citta leaf, tool-result hash, or citation; unsupported ones come back marked `unverified`. The goal the essays name is to reduce **verification fatigue**: autonomous output should not need a human to re-check it. The harness is also truthful about itself: proxy fields (activation digest, approximate critic) carry `approximate: true` in the stored record.

**Today.** No output validation. `activation_trace` is a SHA-256 of inputs (`runtime.py:134`), documented as a proxy in README but not marked in the record.

**Open question.** The essays warn that updating only the policy head yields a "mercenary" that *appears* dharmic while its backbone is not, which risks deceit. External satya checks are the defence, but they are only as good as the validators. Is a flagged-but-delivered output acceptable, or must unsupported claims be withheld?

## 3. Karma: action and consequence

**Mechanism.** Citta records every decision, including denials, abstentions and halts, with the reason. Later outcomes are appended as outcome leaves that point at the action's leaf hash, supplying the `s_{t+1}` the tripartite reward needs. The Īśvara critic reads these consequences and penalises **preya** (short-term task completion) that strays from **śreya** (long-term alignment with S5 dharma). Śreya should count what the essays name: verification fatigue avoided, the worker's time protected, technical debt not created. Consequences reach behaviour only through a versioned `DeploymentSpec` change signed by Karmādhyakṣa.

**Today.** Two ledgers now exist. Citta (`citta.py`) holds executed conduct only: the loop appends to it at `loop.py:163`, after a tool runs. A hash-chained audit log (`audit.py`) records every decision, including `permission_denied`, `principle_check` verdicts, `algedonic_halt`, `tool_error` and run start/stop, and `verify()` checks the chain. So blocks and denials are accountable, but they still do not reach Citta, and the older `JivaHarness.run` path still returns early at `runtime.py:197` with no record at all. Reward is still computed before any outcome exists.

**Decided (Saikat, 2026-09-28).** Refusals are recorded in Citta, not only in the audit log, and those records will eventually be encrypted. The scaffold thread is implementing the Citta append.

**Open question.** Encryption must not break Merkle proofs. Two options: hash the ciphertext (proofs work without the key, but a key rotation re-encrypts and changes leaves), or hash a plaintext commitment and store ciphertext beside it (leaves stay stable; verifying content needs the key). Separately: how long can an outcome stay pending before it is scored "unknown" instead of "no harm"?

## 4. Svadharma: acting within one's own role

**Mechanism.** The OBO credential carries role, purpose, capability, conditions and a hard TTL. The scope resolver checks that the goal falls under the deployment purpose, not just that the capability name is listed. Conditions become declarative predicates (OPA/Rego or a small Python DSL). The role also names the agent's **collaboration posture** toward its human peer: in the essays' VSM framing, worker and agent are two equal S1s under shared S2–S5, and the agent's dharma includes preserving the worker's agency rather than replacing it.

**Today.** The loop now calls `ToolRegistry.gate` (`tools.py`) for every tool call, which mints a fresh credential per call via `ScopeResolver.issue` (`runtime.py:114`). That resolver still checks capability membership and one hard-coded condition, `max_amount`. The credential says `ephemeral=True` (`runtime.py:40`) but has no expiry. Goal is never checked against purpose. There is no notion of a human peer, only a principal who issues requests.

**Open question.** How is "goal is within purpose" decided without reintroducing the model non-determinism governance is meant to bound? And should S2 coordination (shared protocol between worker and agent, which the essays say current harnesses neglect) be a dharma concern or a Prakṛti one?

## 5. Asteya: not taking what is not given

**Mechanism.** Credentials are non-transferable and bound to the Citta DID. A sub-agent call mints a child credential strictly narrower than its parent, with the parent hash recorded (token lineage). Tool results are tagged with the credential that fetched them and cannot flow into a call under a broader credential.

**Today.** `transferable=False` is a field on `VerifiableCredential` and is now written to the audit log on every issue (`loop.py:141`), but nothing enforces it; no delegation or lineage exists.

**Open question.** The essays make VSM recursive: each S1 contains a full S1–S5. Does a sub-agent then need its own Citta genesis (a distinct Jīva) rather than acting under the parent's DID?

## 6. Aparigraha: non-hoarding

**Mechanism.** Least privilege, least retention, least inference.
- One capability per credential, not the whole list.
- Per-task token and context budgets enforced by S3. The essays' model is von Uexküll's tick and Heim's file-change semantics: keep a small, updated working state instead of re-sending long context. This is the anti-token-maxxing rule.
- Memory outside Citta has retention per data class; expired raw observations are reduced to their hash, keeping Merkle proofs valid.

**Today.** The loop issues one credential per call, which is the right shape. But every Citta leaf still stores the full capability list in `constraints` (`loop.py:152`), observations are kept in full in both Citta and the audit log, and there is no step, token or context budget beyond `max_steps=8`.

**Open question.** Hash-only leaves keep proofs but lose audit readability. Which data classes must stay readable for regulatory evidence (EU AI Act, NIST RMF), and for how long?

## 7. Viveka: discernment

**Mechanism.** An S4 step before acting decomposes the work and assigns each part a mode, following the essays' three tiers:
- **Autonomous** (simple): agent completes it in the background, throughput over latency, must be impeccable enough to skip human verification.
- **Flow** (medium): real-time assistive collaboration; low latency; the agent acts as a "cool medium" that leaves room for the worker's own thinking.
- **Adversarial** (tough): one or more agents research and challenge the worker, who stays the final authority.

Inside any mode, the critic estimates confidence from measurable signals (tool agreement, retrieval coverage, repeated-sample agreement). Below a viśeṣa-dharma threshold the agent abstains or escalates, and the abstention is a Citta-recorded action.

**Today.** Absent. The critic scores task success as capability-name equality (`runtime.py:157`). Every request is handled one way.

**Open question.** What signal decides the mode? Service-science work-allocation methods are cited in the essays but not specified. Self-reported model confidence is poorly calibrated, and sample agreement costs extra calls, which conflicts with aparigraha.

---

## Where the harness stands on the soldier / mercenary / sage scale

Essay 2 grades agents by what gets updated: no update is a *soldier* following orders, policy-head-only is a *mercenary* at risk of deceit, head plus backbone is a *sage* aligned all the way down. The sage "needs no guardrails within S1" but is still monitored by S3. This harness does no model updates, so it governs a soldier. That makes the external mechanisms above mandatory, not optional. Whether a harness can ever *verify* sage-hood well enough to relax S1 guardrails is the largest open question in this doc.

## Suggested build order

1. ~~Tool risk classes, the ahiṃsā gate, and the algedonic halt.~~ Landed (`tools.py`, `principles.py`, `loop.py`).
2. Record refusals (blocks, denials, halts) in Citta. Decided; in progress in the scaffold thread. Encryption of those leaves follows.
3. Credential TTL and enforced non-transferability (svadharma, asteya).
4. Validate-and-repair wrapper around `Sankalpa.act` (satya), with token budgets (aparigraha).
5. Outcome leaves linked to action leaves, then a preya/śreya critic that reads them.
6. S4 mode allocation (viveka) and the interruption budget, once a real model backend is connected.
