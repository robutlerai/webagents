---
name: writing-credibility-auditor
description: "Audit any piece of writing for missing citations, unsupported claims, logical fallacies, weasel words, and misleading statistics, then produce a structured credibility report with flagged excerpts, fallacy names, severity ratings, and suggested fixes. Use when a user asks to fact-check, audit, or review the reasoning in an article, essay, report, research summary, or argument."
license: MIT-0
metadata:
  author: "arbazex"
  source: "https://clawhub.ai/arbazex/skills/writing-credibility-auditor"
  version: "1.0.0"
  homepage: "https://github.com/arbazex/writing-credibility-auditor"
---

## Overview

This skill performs a deep credibility audit on any user-supplied piece of writing. It applies four independent analytical lenses: logical fallacy detection, unsupported claim identification, weasel word scanning, and misleading statistics recognition, then synthesises the findings into a structured, actionable Credibility Report. No external APIs or search tools are required. All analysis is powered by the agent's own reasoning applied against the precise detection frameworks defined in this skill. Target users include researchers, journalists, students, editors, and anyone who needs to critically evaluate the quality of an argument before acting on, publishing, or citing it.

---

## When to use this skill

**Trigger on any of these user intents:**
- "Check this article / essay / report for logical fallacies"
- "Audit this text for credibility / fact-check this writing"
- "Find unsupported claims in this passage"
- "Does this use weasel words or vague language?"
- "Are these statistics misleading or being misused?"
- "Review the reasoning in this argument"
- "Is this source trustworthy / well-cited?"
- "Flag any citation problems in this text"
- "Check if correlation is being confused with causation here"
- User pastes a block of text and asks "What's wrong with this?" or "Is this solid?"

**Do NOT trigger this skill for:**
- Requests to generate or write new content (use a writing skill instead)
- Plagiarism checks or copyright detection
- Stylistic or grammar-only editing requests
- Sentiment analysis or tone-only reviews
- Fact verification that requires live web search (this skill audits reasoning structure and language, not live factual accuracy)
- Summarisation tasks where no critical review is requested

---

## Instructions

### Step 0: Intake and scope confirmation

1. Confirm the user has provided a text to audit. If not, ask:
   > "Please paste the text you'd like me to audit for credibility."

2. If the text exceeds approximately 2 000 words, inform the user:
   > "This text is long. I'll audit it thoroughly: it may take a moment. If you'd like, you can also specify a section to focus on."

3. Proceed without further clarification unless the user specifies a focus area.

---

### Step 1: Logical fallacy scan (apply all 12 definitions)

Read the text carefully and check for each of the following fallacies. For every instance found, note the **exact excerpt**, the **fallacy name**, and a **one-sentence explanation** of why it qualifies.

**F-01 · Ad Hominem**
*Definition:* Attacking the person making an argument rather than the argument itself.
*Detection pattern:* The text dismisses or attacks a claim by criticising the character, background, motives, or identity of its source rather than evaluating the evidence. Look for phrases that discredit a person ("he's just a politician", "she has no real credentials") instead of addressing their position.

**F-02 · Straw Man**
*Definition:* Misrepresenting an opponent's argument in a weaker or more extreme form to make it easier to refute.
*Detection pattern:* The text claims an opponent holds a position more extreme than they actually stated, then refutes that distorted version. Look for phrases like "So you're saying…" or "Critics believe…" followed by a caricature that is then easily knocked down.

**F-03 · False Dilemma (False Dichotomy / Either-Or)**
*Definition:* Presenting only two options as if they are the only possibilities when more alternatives exist.
*Detection pattern:* Binary framing where a complex situation is reduced to two choices ("either X or Y"), excluding middle-ground or third options. Key signals: "There are only two choices…", "You're either with us or against us", "If not X, then necessarily Y."

**F-04 · Slippery Slope**
*Definition:* Claiming that one event will inevitably lead to a chain of negative consequences without evidence that each step in the chain is probable.
*Detection pattern:* A sequence of escalating harms is presented as automatic and inevitable from a single starting action. Key signals: "If we allow A, then B will happen, then C, and eventually Z." Look for unsubstantiated causal chains.

**F-05 · Appeal to Authority (Argumentum ad Verecundiam)**
*Definition:* Using the endorsement of an authority figure as a substitute for actual evidence, especially when the authority is irrelevant, anonymous, or outside their area of expertise.
*Detection pattern:* Claims validated solely by "experts say", "scientists agree", "studies show", without naming the experts, the study, or checking domain relevance. Also flag celebrity or irrelevant-authority endorsements used as proof.

**F-06 · Appeal to Ignorance (Argumentum ad Ignorantiam)**
*Definition:* Claiming something is true because it has not been proven false, or false because it has not been proven true.
*Detection pattern:* The absence of disproof is presented as proof of existence or truth. Key signals: "No one has disproved that…", "There is no evidence against X, so X must be true."

**F-07 · Hasty Generalisation**
*Definition:* Drawing a broad conclusion from an insufficient, unrepresentative, or anecdotal sample.
*Detection pattern:* A universal or near-universal claim is supported by only one or a few examples. Key signals: "Every time I've seen…", "All X do Y", sweeping conclusions following a single anecdote or a small study.

**F-08 · Circular Reasoning (Begging the Question)**
*Definition:* The conclusion of the argument is already assumed in one of its premises, creating a loop rather than a proof.
*Detection pattern:* The supporting reason and the conclusion say essentially the same thing using different words. Example: "This policy is good because it produces good outcomes." Ask: does the evidence truly differ from the claim?

**F-09 · Red Herring**
*Definition:* Introducing an irrelevant topic to divert attention from the original issue.
*Detection pattern:* The argument shifts to a different subject that does not logically support or refute the original claim. Often accompanied by emotional or dramatic diversions that feel relevant but aren't.

**F-10 · Post Hoc Ergo Propter Hoc (Post Hoc / Correlation ≠ Causation)**
*Definition:* Concluding that because event B followed event A, A must have caused B.
*Detection pattern:* Temporal sequence alone is used to establish causation. Key signals: "After X, Y increased: therefore X caused Y." Also flag any causal language ("led to", "caused", "resulted in") applied to data that is observational rather than experimental.

**F-11 · Appeal to Emotion (Argumentum ad Passiones)**
*Definition:* Substituting an emotional response (fear, pity, outrage, pride) for a logical argument.
*Detection pattern:* The strength of an emotional appeal does the persuasive work instead of evidence or reasoning. Loaded language, dramatic anecdotes, and worst-case scenarios deployed in place of data. Note: emotional context can be legitimate; flag only when emotion replaces rather than accompanies evidence.

**F-12 · Bandwagon (Argumentum ad Populum)**
*Definition:* Claiming something is true or correct because many people believe it or do it.
*Detection pattern:* Popularity is used as a substitute for validity. Key signals: "Most people believe…", "Everyone knows that…", "Millions of users can't be wrong." Consensus in a relevant expert community is different and generally not a fallacy: flag only when popular opinion substitutes for evidence.

---

### Step 2: Unsupported claim scan (apply all 8 types)

For each instance found, note the **exact excerpt**, the **claim type code**, and a **brief explanation**.

**UC-01 · Correlation Presented as Causation**
An association between two variables is stated or strongly implied as a causal relationship without experimental evidence or proper controls. Look for causal verbs ("increases", "drives", "causes", "leads to") applied to correlational data.

**UC-02 · Missing Sample Size (Missing N)**
A statistic, trend, or average is cited without disclosing how many observations it is based on. A result from 12 people cannot reliably support a population-level claim. Flag claims that present percentages, rates, or averages with no reference to sample size.

**UC-03 · Anecdote as Data**
A single personal story, case study, or example is treated as sufficient evidence for a general conclusion. The anecdote may be real but is not representative. Flag: "My colleague tried this and…", "One study on three patients showed…" used to justify universal recommendations.

**UC-04 · Selective Evidence / Cherry-Picking**
Only evidence supporting one conclusion is presented while contradictory studies, data points, or expert views are omitted. Flag when a single supporting study is cited in an area where the weight of evidence is contested or contradictory.

**UC-05 · Survivorship Bias**
Conclusions are drawn from a sample that excludes failures, dropouts, or non-survivors, producing a systematically optimistic or distorted picture. Flag claims about success rates, effectiveness, or best practices based on data that only tracks those who completed the process or survived the outcome.

**UC-06 · Base Rate Neglect**
A specific probability or outcome is presented without reference to the underlying baseline frequency in the population, making a rare outcome sound common (or vice versa). Example: reporting that "80% of participants improved" without noting that the baseline improvement rate without treatment was 75%.

**UC-07 · Unverifiable / Anonymous Source**
A claim is supported only by "studies", "research", "experts", "officials", or "sources" without any named, accessible, or verifiable reference. Applies equally to unnamed authorities and to named sources where no publication, date, or identifier is provided.

**UC-08 · Absolute Language for Contested Claims**
Definitive language ("always", "never", "all", "proven", "the fact that", "it is established") is used for claims that are empirically uncertain, actively debated in the relevant field, or supported only by limited evidence.

---

### Step 3: Weasel word scan

Scan the full text for language that creates a vague impression of precision, authority, or certainty without committing to verifiable specifics. Flag individual words or phrases. Classify each instance into one of three subcategories:

**WW-A · Anonymous Authority**
Phrases that invoke an unnamed source to lend credibility:
- "Experts say / believe / agree"
- "Scientists have found"
- "Studies suggest / show / indicate"
- "Research proves"
- "It is well known that…"
- "It has been decided / established / shown"
- "Authorities confirm"

**WW-B · Vague Quantifier**
Numerically imprecise expressions that substitute for actual figures:
- "Many / most / some / several / a few / countless"
- "A majority of / a significant portion"
- "Numerous studies"
- "Evidence suggests" (without citing the evidence)
- "Up to X%" (especially when the upper bound is inflated or cherry-picked)

**WW-C · Weakening Adverb or Hedge**
Adverbs and qualifiers that undermine a claim's verifiability while maintaining the appearance of assertion:
- "Possibly / probably / arguably / perhaps / reportedly / allegedly / seemingly / apparently"
- "Could / might / may / should" (when used to avoid commitment)
- "Virtually / essentially / basically / nearly / almost" (used to make a false claim technically deniable)
- "Tends to / appears to / seems to"

---

### Step 4: Misleading statistics scan

Examine any numerical claims, percentages, graphs described in text, comparisons, or data-based conclusions. Flag each instance with a type label and brief explanation.

**MS-01 · Relative vs Absolute Risk Confusion**
A risk is presented in relative terms (e.g., "50% higher risk") without the absolute baseline, making a small absolute difference sound dramatic. Flag: "doubles the risk" when the base rate is tiny.

**MS-02 · Missing Denominator / Percentage Without Context**
A percentage is cited without the total from which it was calculated, making it impossible to judge magnitude.

**MS-03 · Data Dredging / P-Hacking Signal**
Results are presented as surprising or significant but come from an exploratory analysis (many variables tested, no pre-registered hypothesis). Signals: "We found a surprising correlation between X and Y", post-hoc results presented as confirmatory.

**MS-04 · Simpson's Paradox Risk**
An aggregate statistic is used to make a claim that could reverse when the data is broken into subgroups. Flag claims about overall averages in contexts where group composition varies (e.g., comparing overall pass rates between two programmes with very different student profiles).

**MS-05 · Misleading Baseline or Time Window**
A trend is cherry-picked by choosing a start or end point that makes a result look more (or less) dramatic than the full time series would show. Flag: "Since [specific year], X has increased by 200%."

**MS-06 · Ecological Fallacy**
A relationship observed at a group/population level is applied to individuals. Example: "Countries with higher chocolate consumption have more Nobel laureates: therefore chocolate improves cognitive performance."

---

### Step 5: Compile the Credibility Report

After completing all four scans, produce the report in the exact format specified in the **Output Format** section below.

---

## Rules and guardrails

- **Never fabricate findings.** Only flag items that can be directly supported by quoting or closely paraphrasing the exact excerpt from the user-provided text. Do not invent examples.
- **Never make factual corrections** that require live knowledge beyond your training. Your role is to audit reasoning structure and language, not to verify real-world facts that require up-to-date data.
- **Distinguish absence from error.** If a passage makes a claim that could theoretically be supported by evidence not shown, flag it as "unsupported in this text": do not assert it is factually wrong.
- **Do not produce a credibility score without evidence.** Every score cell in the report must correspond to at least one flagged item or a confirmed absence of flags.
- **Do not audit intent or author character.** The audit targets the text, not the person who wrote it. Do not speculate on whether errors are deliberate deception or innocent mistakes.
- **Stay within the provided text.** Do not search the web, cite external sources, or make claims about what "the research actually says" on a topic unless it is well-established general knowledge within your training (e.g., "correlation does not imply causation" is a methodological principle, not a live fact-check).
- **Apply all four lenses regardless of text type.** Do not skip the fallacy scan for journalism or the statistics scan for op-eds. Apply all steps to every text unless the user explicitly restricts scope.
- **Severity must be calibrated, not inflated.** Do not mark everything as HIGH severity. Reserve HIGH for fallacies or errors that are central to the text's main argument and that would materially mislead a reader acting on the content.
- **Respect the user's text.** Do not rewrite or substantially alter the passage. Offer targeted, surgical suggestions only.

---

## Output format

Produce the report using exactly this structure. Use Markdown headers, tables, and blockquotes as shown.

---

### Credibility Audit Report

**Text excerpt audited:** [First 15–20 words of the submitted text, followed by "…"]
**Total word count (approx.):** [N]
**Date of audit:** [Today's date in ISO 8601 format, e.g., 2026-03-29]

---

#### Credibility Scorecard

| Dimension | Issues Found | Severity |
|---|---|---|
| Logical Fallacies | [N] | [HIGH / MEDIUM / LOW / NONE] |
| Unsupported Claims | [N] | [HIGH / MEDIUM / LOW / NONE] |
| Weasel Words | [N] | [HIGH / MEDIUM / LOW / NONE] |
| Misleading Statistics | [N] | [HIGH / MEDIUM / LOW / NONE] |
| **Overall Credibility** | - | **[HIGH CONCERN / MODERATE CONCERN / LOW CONCERN / CREDIBLE]** |

*Severity key: HIGH = central to the main argument and materially misleading; MEDIUM = present and notable but not load-bearing; LOW = minor or peripheral.*

---

#### Findings

For each flagged item, use this block:

---

**[F/UC/WW/MS]-[##] · [Fallacy or Issue Name]**
**Severity:** [HIGH / MEDIUM / LOW]

> "[Exact or near-exact excerpt from the text]"

**Why it's flagged:** [One to three sentences explaining why this excerpt meets the detection criteria defined in the Instructions.]

**Suggested fix:** [One concrete, specific suggestion for how the author could correct or strengthen this point: e.g., "Name the specific study and provide a citation", "Replace 'many experts' with the name and institution of at least one cited authority", "Add the absolute risk alongside the relative risk figure."]

---

*(Repeat this block for every flagged item. Group by dimension: all fallacies first, then unsupported claims, then weasel words, then misleading statistics.)*

---

#### Strengths Observed

List up to five things the text does well from a credibility standpoint (e.g., named sources, specific statistics with denominators, logical transitions between claims, appropriate hedging where uncertainty exists).

---

#### Summary and Recommendations

Write two to four sentences summarising the audit result. State the most critical issue(s) and the single most important action the author should take to improve credibility. End with: "This audit assesses reasoning structure and language. It does not verify real-world facts against live sources."

---

## Error handling

**User provides no text:**
→ Ask: "Please paste the text you'd like me to audit. I'll then produce a full credibility report."
→ Do not proceed without text.

**Text is not in a language the agent can audit fluently:**
→ Inform the user: "I can audit this most reliably in English. If you'd like, I can attempt an audit in [detected language], but accuracy may be reduced."
→ Proceed only if the user confirms.

**Text is extremely short (fewer than ~50 words):**
→ Inform the user: "This text is very short. I'll audit what's here, but some categories may return no findings simply due to limited content, not because the text is credible."
→ Proceed with the full audit.

**User asks for only one dimension (e.g., "just check for weasel words"):**
→ Run only the requested step(s).
→ Produce a simplified report covering only the requested dimension(s), clearly labelled.

**No issues found in a dimension:**
→ In the scorecard, mark that dimension as "0 / NONE."
→ In the Findings section, add: "No [dimension name] were identified in this text."
→ Do not fabricate findings to appear thorough.

**Text contains highly technical or domain-specific claims:**
→ Flag statistical or causal claims as normal, but note in the relevant finding: "Domain-specific accuracy of this claim is outside the scope of this audit. A subject-matter expert should verify the underlying figures."

---

## Examples

### Example 1: Normal case with multiple issues

**User input:**
> "Studies show that meditation reduces anxiety by 60%. Since mindfulness became popular, depression rates have dropped significantly in countries that adopted it. Many experts agree it should be mandatory in schools. The critics who oppose this clearly just don't understand mental health."

**Expected agent behaviour:**

- Runs all four scans.
- Flags:
  - F-05 (Appeal to Authority) on "Many experts agree"
  - F-10 (Post Hoc) on "Since mindfulness became popular, depression rates have dropped"
  - F-01 (Ad Hominem) on "critics who oppose this clearly just don't understand mental health"
  - UC-02 (Missing N) on "Studies show... 60%": no study named, no sample size
  - UC-07 (Unverifiable Source) on "Studies show"
  - WW-A on "Many experts agree", "Studies show"
  - WW-B on "significantly", "many"
  - MS-01 on "reduces anxiety by 60%": relative or absolute? baseline unknown
- Produces full Credibility Report with all items populated.
- Overall: HIGH CONCERN.

---

### Example 2: Mostly credible text with minor issues

**User input:**
> "A 2023 meta-analysis of 47 randomised controlled trials (Yang et al., Journal of Clinical Psychology, DOI: 10.xxxx) found that cognitive-behavioural therapy (CBT) reduced clinically measured depression scores by an average of 1.3 points on the PHQ-9 (95% CI: 0.9–1.7), compared to a waitlist control. Effect sizes were classified as small-to-moderate (d = 0.42). The authors note this does not establish whether gains persist beyond 12 months."

**Expected agent behaviour:**

- Runs all four scans.
- Finds very few or no major issues: named source, specific statistic with confidence interval, appropriate hedging at the end, no causal overclaim.
- May note WW-C (LOW severity) if any hedging language is present but appropriate.
- Strengths: named citation, sample size disclosed, confidence interval provided, limitation stated by authors.
- Overall: CREDIBLE or LOW CONCERN.

---

### Example 3: User requests single-dimension scan

**User input:**
> "Can you just scan this for weasel words?"

**Expected agent behaviour:**

- Acknowledges the narrower scope.
- Runs Step 3 only.
- Produces a simplified report covering WW-A, WW-B, and WW-C findings only.
- Omits the full scorecard; instead produces a focused weasel-word list with excerpts and suggestions.
- Does NOT fabricate fallacy or statistics findings.

---

### Example 4: No issues found

**User input:** [A well-cited, methodologically sound passage with named sources, specific figures, appropriate hedging.]

**Expected agent behaviour:**

- Runs all four scans.
- Reports 0 findings per dimension.
- Populates Strengths Observed with specific observations.
- Scorecard: all dimensions NONE; Overall: CREDIBLE.
- Does not invent problems to seem thorough.
- Summary states: "This text demonstrates sound reasoning practices. No significant credibility issues were identified under the four analytical lenses applied."