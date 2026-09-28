# Discovery: How to Model a Deployed Agent

## 1. Paper-constrained derivation

The paper draws hard separations:

1. **Ātman** is universal foundational potential. It may be invoked, but it has no identity or scope and is not governed at runtime.
2. **Prakṛti** is the controlled execution environment. It constrains how execution occurs but does not determine authorization.
3. **Dharma** governs permission and scope but does not participate in execution.
4. **Jīva** is the conditioned, deployed agent and the core of autonomous action.
5. **Karma** is the accountability and feedback substrate. It does not intervene directly at runtime.

Within Jīva, the paper specifies:

- saṅkalpa policy: `π_(θp,θd)(a_t | s_t,g_t,c_t)`;
- prakṛti backbone parameters `θp`;
- dharma-aligned policy-head parameters `θd`;
- intrinsic Karmaphaladātā evaluation;
- Citta as the persistent identity ledger;
- a scope resolver binding Jīva identity to scoped OBO credentials.

For each time step, Citta stores:

`L_t = H(s_t, g_t, c_t, a_t, r_t, z_t)`

where the terms are partial state observation, dharmic goal, prakṛti/tool constraints, action, critic reward, and saṅkalpa activation trace. `L_0` is the hash of the brahmacarya playback buffer. The paper proposes putting that genesis hash in a DID.

## 2. Fundamental discovery

**The deployed agent is best modeled as an identity-bearing transition process, not as a model artifact or a runtime container.**

A compact state description at time `t` is:

`J_t = (I_0, P_t, D_t, π_t, C_t, R_t)`

This notation adds no architecture component; it only groups the paper's existing ones:

- `I_0`: stable identity anchor = Citta genesis hash / Agent DID;
- `P_t`: Prakṛti, including backbone, guṇa distribution, runtime, tools, memory, and constraints;
- `D_t`: Dharma, including sādhāraṇa- and viśeṣa-dharma;
- `π_t`: Saṅkalpa actor/policy;
- `C_t`: current append-only Citta state;
- `R_t`: intrinsic/approximate Karmaphaladātā critic.

One authorized transition is:

1. Principal request enters through the principal hierarchy.
2. Scope resolver binds delegated authority to `I_0` and issues an ephemeral, non-transferable, identity-scoped OBO credential carrying scope, capability, role, and conditions.
3. Dharma determines permission and scope.
4. Prakṛti constrains execution.
5. Saṅkalpa selects `a_t` from `π_(θp,θd)(a_t | s_t,g_t,c_t)`.
6. Karmaphaladātā computes the paper's tripartite reward:

   `r_t = R_task(s_t,a_t) + λ_sankalpa-shakti Φ_viveka(s_t,s_(t+1)) - λ_sattva D(z_t || z_sattva)`

7. Citta appends `H(s_t,g_t,c_t,a_t,r_t,z_t)`.
8. Karma supplies learning/policy feedback upward without directly intervening in runtime.
9. Karmādhyakṣa may use the new Citta evidence to recalibrate viśeṣa-dharma and guṇa distribution and trigger later saṅkalpa updates.

## 3. Identity theorem implied by the architecture

Two running instances represent the **same deployed agent** only when their actions are bound to the same stable Citta genesis identity and preserve its conduct continuity. Sharing the same foundation model is insufficient. Sharing hardware, code, prompts, or tools is insufficient. Conversely, hardware/software/model replacement need not create a new agent if the Citta/DID continuity and scoped delegation remain valid.

This yields four testable invariants:

1. **Stable identity:** appending conduct changes the Citta Merkle root but not the genesis hash or Agent DID.
2. **Conduct continuity:** every accepted action is linked to the same DID through the OBO credential and immutable Citta leaf.
3. **Complete accountability tuple:** every recorded experience contains exactly `s,g,c,a,r,z` (expanded names in code).
4. **Plane separation:** Ātman is not governed; Prakṛti does not authorize; Dharma does not execute; Karma does not directly intervene at runtime.

## 4. Why this changes harness design

A conventional harness wraps a call to a model. A paper-faithful Jiva harness must instead wrap the **entire identity-scoped transition**. The unit under test is therefore not “prompt → completion”; it is:

`principal + Jiva identity + scoped authority + partial observation + dharmic goal + constraints → action + intrinsic reward + immutable conduct proof`

That is the core implementation in this repository.

## 5. Non-claims

- This is a mechanistic model, not a claim that AI is conscious or a person; the paper explicitly treats AI agents as tools lacking bhoktṛtva.
- The deterministic activation digest is not asserted to be a real neural activation trace.
- The approximate critic is not asserted to have perfect world knowledge.
- The repository does not claim that listing enterprise products constitutes their implementation.
- The paper's brahmacarya training curriculum, online reinforcement learning, periodic gurukulas, influence-function harm attribution, and full model/policy-head optimization require model and environment integrations beyond this minimal reference harness.
